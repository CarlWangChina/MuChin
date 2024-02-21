import os
import torch
import torchaudio
from torch.utils.data import Dataset
from einops import rearrange
import logging
import shutil
import shlex
import json
import numpy
import traceback
from pathlib import Path
import subprocess
import pyloudnorm as pyln

def run_cmd_sync(cmd, cwd=None, interactive=False, timeout=None):
    """Runs a console command synchronously and returns the results.

    Parameters
    ----------
    cmd : str
       The command to execute.
    cwd : :class:`pathlib.Path`, optional
       The working directory in which to execute the command.
    interactive : bool, optional
       If set, run command interactively and pipe all output to console.
    timeout : float, optional
       If specified, kills process and throws error after this many seconds.

    Returns
    -------
    int
       Process exit status code.
    str, optional
       Standard output (if not in interactive mode).
    str, optional
       Standard error (if not in interactive mode).

    Raises
    ------
    :class:`FileNotFoundError`
       Unknown command.
    :class:`NotADirectoryError`
       Specified working directory is not a directory.
    :class:`subprocess.TimeoutExpired`
       Specified timeout expired.
    """
    if cmd is None or len(cmd.strip()) == 0:
        raise FileNotFoundError()

    kwargs = {}
    if not interactive:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    err = None
    with subprocess.Popen(shlex.split(cmd), cwd=cwd, **kwargs) as p:
        try:
            p_res = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            err = e
        p.kill()

    if err is not None:
        raise err

    result = p.returncode

    if not interactive:
        stdout, stderr = [s.decode("utf-8").strip() for s in p_res]
        result = (result, stdout, stderr)

    return result

def get_approximate_audio_length(path, timeout=10):
    """Retrieves the approximate length of an audio file."""
    status, stdout, stderr = run_cmd_sync(
        f"ffprobe -v error -i '{path}' -show_format -show_streams -print_format json",
        timeout=timeout,
    )
    try:
        assert status == 0
        assert len(stderr) == 0
    except:
        raise Exception(f"FFmpeg failed: {stderr}({path})")
    d = json.loads(stdout)
    duration = float(d["format"]["duration"])
    return duration

def normalize_loudness(audio: torch.Tensor,
                       sampling_rate: int,
                       target_loudness: float) -> torch.Tensor:
    """Normalize loudness.

    Args:
        audio (torch.Tensor):           The audio tensor.  Shape: (num_channels, num_samples), or (num_batches,
                                        num_channels, num_samples).
        sampling_rate (int):            The sampling rate.
        target_loudness (float):        The target loudness in dB LUFS.

    Returns:
        torch.Tensor:                   The normalized audio tensor.  Shape is same to the audio.
    """
    assert audio.dim() in [2, 3], "Unsupported audio shape: {}.  Should be 2 or 3".format(audio.shape)

    if audio.dim() == 3:
        audio_list = [
            AudioUtils.normalize_loudness(audio[i], sampling_rate, target_loudness)
            for i in range(audio.shape[0])
        ]
        return torch.stack(audio_list, dim=0)

    meter = pyln.Meter(sampling_rate)
    audio_npd = rearrange(audio, "c n -> n c").numpy()
    loudness = meter.integrated_loudness(audio_npd)
    audio_normalized = pyln.normalize.loudness(audio_npd, loudness, target_loudness)

    return rearrange(torch.from_numpy(audio_normalized), "n c -> c n")

class AudioDataset(Dataset):
    def __init__(self,
                 pathes: list[str],
                 rootPath: str,
                 outputPath:str,
                 statePath:str,
                 rank: int = 0,
                 getLength: bool = False,
                 output_group_size=10000):
        super(AudioDataset, self).__init__()
        self.rank = rank
        self.output_group_size = output_group_size
        self.getLength = getLength

        self.song_pathes = []

        for path in pathes:
            path = os.path.abspath(path)
            rootPath = os.path.abspath(rootPath)
            assert os.path.commonpath([path, rootPath]) == rootPath , f"{path} is not in {rootPath}."

            relative_path = os.path.relpath(path, rootPath)
            absolute_path = os.path.abspath(os.path.join(outputPath, relative_path))
            state_path = os.path.abspath(os.path.join(statePath, relative_path))
            
            if not os.path.exists(state_path):
                self.song_pathes.append((path, absolute_path, state_path))
            # else:
            #     print("skip:", path)

    def __len__(self) -> int:
        return len(self.song_pathes)

    def __getitem__(self, idx):
        file_path_pair = self.song_pathes[idx]
        file_path = Path(file_path_pair[0])
        out_path = Path(file_path_pair[1])
        state_path = Path(file_path_pair[2])
        
        assert file_path.exists(), f"Data path {file_path} does not exist."
        assert file_path.is_file(), f"Data path {file_path} is not a file."

        error = None

        audio_length = 0

        try:
            audio, sampling_rate = torchaudio.load(file_path)
            audio = normalize_loudness(audio, sampling_rate, -12)
            if self.getLength:
                audio_length = get_approximate_audio_length(file_path)
        except Exception as e:
            error = e
            print(f"Failed to load file {file_path}: {str(e)}")
            audio = None
            sampling_rate = None

        if error is None:
            return {
                "file_path": file_path,
                "out_path": out_path,
                "state_path": state_path,
                "audio_length": audio_length,
                "file_group": idx // self.output_group_size,
                "audio_data": audio,
                "sampling_rate": sampling_rate
            }
        else:
            return {
                "file_path": file_path,
                "out_path": out_path,
                "error": error
            }

    @staticmethod
    def collate_fn(batch: list):
        assert len(batch) == 1
        return batch[0]

class VecDataset(Dataset):
    def __init__(self, path:str, output_len:int, data_count:int=4500, data_start:int=0, mean_dim:int=0, preload:bool=True):
        super(VecDataset, self).__init__()
        self.datas = []
        self.output_len = output_len
        self.mean_dim = mean_dim
        self.data_count = data_count
        self.index_count = torch.ones(output_len)
        self.return_path = False
        count = 0
        with open(path) as fp:
            line = fp.readline()
            while line:
                line = line.strip()
                if line!="":
                    data = json.loads(line)
                    try:
                        if os.path.exists(data["path"]):
                            assert "tag" in data
                            content = None
                            if preload:
                                content = self.loadTensorFromFile(data["path"])
                                print(data["path"],count)
                            if count>=data_start:
                                self.datas.append({"path":data["path"],"tag":data["tag"],"data":content})
                            self.index_count += buildOutputTensor(output_len, data["tag"])
                            count += 1
                            if count>=self.data_count:
                                break
                    except FileNotFoundError as err:
                        print(err,data)
                line = fp.readline()
        self.index_count_max = self.index_count.max()
        self.weight = self.index_count_max/self.index_count
        # print(self.index_count_max)
        # print(self.index_count)
        # print(self.weight)

    def __len__(self) -> int:
        return len(self.datas)

    def loadTensorFromFile(self, path:str):
        return torch.load(path, map_location="cpu")["data"].mean(dim=self.mean_dim)

    def __getitem__(self, idx:int):
        if not self.datas[idx]["data"] is None:
            input_data = self.datas[idx]["data"]
        else:
            input_data = self.loadTensorFromFile(self.datas[idx]["path"])
        output_data = buildOutputTensor(self.output_len, self.datas[idx]["tag"])
        if self.return_path:
            return input_data, output_data, self.datas[idx]["path"]
        else:
            return input_data, output_data

class VecDatasetReplace(VecDataset):
    def __init__(self, path:str, output_len:int, data_count:int=4500, data_start:int=0, mean_dim:int=0, preload:bool=True, replace:list=None):
        self.replace = replace
        super(VecDatasetReplace, self).__init__(path, output_len, data_count, data_start, mean_dim, preload)
    def loadTensorFromFile(self, path:str):
        if not self.replace is None:
            for it in self.replace:
                path = path.replace(it["src"], it["tgt"])
        return torch.load(path, map_location="cpu")["data"].mean(dim=self.mean_dim)

class VecDatasetSearch(VecDataset):
    def __init__(self, path:str, output_len:int, data_count:int=4500, data_start:int=0, mean_dim:int=0, preload:bool=True, replace:list=None, dir_path:str="./"):
        self.scanDirs = []
        self.replace = replace
        # print(self.replace)
        self.scanDir(dir_path)
        super(VecDatasetSearch, self).__init__(path, output_len, data_count, data_start, mean_dim, preload)
    def loadTensorFromFile(self, path:str):
        if not self.replace is None:
            for it in self.replace:
                path = path.replace(it["src"], it["tgt"])
        # print(path,self.replace)
        path = findFileByName(os.path.basename(path), self.scanDirs)
        return torch.load(path, map_location="cpu")["data"].mean(dim=self.mean_dim)
    def scanDir(self, path:str):
        for entry in os.scandir(path):    
            if entry.is_dir():
                filename = os.path.abspath(entry.path)  
                self.scanDirs.append(filename)
        # print(self.scanDirs)


class VecDatasetMusic2Vec(VecDatasetReplace):
    def loadTensorFromFile(self, path:str):
        if not self.replace is None:
            for it in self.replace:
                path = path.replace(it["src"], it["tgt"])
        data = torch.load(path, map_location="cpu")
        return data.mean(dim=self.mean_dim)

class VecDatasetJukeBox(VecDatasetReplace):
    def loadTensorFromFile(self, path:str):
        if not self.replace is None:
            for it in self.replace:
                path = path.replace(it["src"], it["tgt"])
        data = torch.load(path, map_location="cpu")
        # print(data)
        return data.mean(dim=self.mean_dim)

class VecDatasetEncodec(VecDatasetReplace):
    def loadTensorFromFile(self, path:str):
        if not self.replace is None:
            for it in self.replace:
                path = path.replace(it["src"], it["tgt"])
        data = torch.load(path, map_location="cpu").view(-1)
        return data


def findFileByName(file_name:str, search_paths:list)->str:  
    for path in search_paths:  
        if os.path.exists(os.path.join(path, file_name)):  
            return os.path.join(path, file_name)  
        if os.path.exists(os.path.join(path, file_name+".gz")):  
            return os.path.join(path, file_name)  
    raise FileNotFoundError(f"File {file_name} not found in any of the paths: {search_paths}")

def buildOutputTensor(output_size:int, one_index:list[int]):
    res = numpy.zeros([output_size])
    for i in one_index:
        res[i] = 1.0
    return torch.tensor(res.astype(numpy.float32))

def get_dir_pathes(input_dir:str):
    file_list = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list
