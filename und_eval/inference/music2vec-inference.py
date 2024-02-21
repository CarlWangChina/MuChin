import sys
sys.path.append("./")

import models.music2vec
import dataset
import os
import json
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import logging
import shutil
from pathlib import Path
from tqdm import tqdm

from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/models-inference.yaml')

host = {
  "dist_backend": "nccl",
  "master_addr": "127.0.0.1",
  "master_port": 25683
}

input_dir = cfg.model.music2vec.input_dir
output_dir = cfg.model.music2vec.output_dir
state_dir = cfg.model.music2vec.state_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _setup(rank: int,
           host: dict,
           world_size: int):
    os.environ['MASTER_ADDR'] = host["master_addr"]
    os.environ['MASTER_PORT'] = str(host["master_port"])

    # initialize the process group
    dist.init_process_group(backend=host["dist_backend"],
                            rank=rank,
                            world_size=world_size)

def _cleanup():
    dist.destroy_process_group()

def checkDir(file_path):
    # 从文件路径中提取父目录  
    parent_directory = os.path.dirname(file_path)  
      
    # 检查目录是否存在  
    if not os.path.exists(parent_directory):  
        # 如果目录不存在，创建它  
        os.makedirs(parent_directory)

pathes = []
with open("/nfs/datalists/s1_ans_4mue-240122_103712_lo.jsonl") as fp:
    line = fp.readline()
    while line:
        arr = json.loads(line)
        path = arr["audio_lp"]
        pathes.append(path)

        relative_path = os.path.relpath(path, input_dir)
        checkDir(os.path.abspath(os.path.join(output_dir, relative_path)))
        checkDir(os.path.abspath(os.path.join(state_dir, relative_path)))

        line = fp.readline()

def processFile(enc, batch):
    if os.path.exists(batch["state_path"]):
        return

    audio = batch["audio_data"]

    if audio.dim()==2:
        audio = audio.mean(dim=0)

    res = enc.encode(audio,batch["sampling_rate"])
    # print(res)
    torch.save(res, batch["out_path"].with_suffix(".pkl"))
    with open(batch["state_path"], "w") as fp:
        pass

def process(rank:int, world_size:int, host:dict, data_loader_num_workers:int=2):
    torch.cuda.set_device(rank)
    _setup(rank, host, world_size)
    
    device = f"cuda:{rank}"

    ds = dataset.AudioDataset(pathes=pathes,
                              rank=rank,
                              rootPath=input_dir,
                              statePath=state_dir,
                              outputPath=output_dir,
                              getLength = False)
        
    data_sampler = DistributedSampler(ds,
                                      num_replicas=world_size,
                                      rank=rank)

    data_loader = torch.utils.data.DataLoader(ds,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=data_loader_num_workers,
                                              collate_fn=dataset.AudioDataset.collate_fn,
                                              sampler=data_sampler)
    
    enc = models.music2vec.music2vecEncoder(device=device)
    
    for batch in tqdm(data_loader, desc=f"Rank {rank}"):
        if "error" not in batch:
            processFile(enc, batch)

    _cleanup()
    

if __name__ == "__main__":


    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        logger.warning("CUDA_VISIBLE_DEVICES is not set.  Using all available GPUs.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()

        logger.info(f"Using {world_size} CUDA GPUs.")

        torch.multiprocessing.spawn(process,
                                    args=(world_size, host),
                                    join=True,
                                    nprocs=world_size)