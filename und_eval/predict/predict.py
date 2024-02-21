import sys
sys.path.append("./")

import torch  
import json  
from models.mlp import MLP ,MLPEmbedding ,F1_score
from dataset import *
from tqdm import tqdm  
from ans.QuestionEncoder import EncoderGroup

decoder_s0 = EncoderGroup.load_from_file("../datas/s0_ans_dict.txt")
decoder_s1 = EncoderGroup.load_from_file("../datas/s1_ans_dict.txt")
device = "cuda:4"

class ModelScore:
    def __init__(self, theta):
        self.theta = theta
    
    def make_global_score(self):
        self.no_outputs_count = []
        self.data_count = 0
        for i in range(10):
            self.no_outputs_count.append(0)
    
    def save_global_score(self, method):
        with open(f"outputs/{self.theta}/score-{method}.txt", "w") as score_fp:
            for i in range(10):
                score_fp.write(f"{i}\tno_outputs_count:{self.no_outputs_count[i]}/{self.data_count}\n")
            sum_no_out = sum(self.no_outputs_count)
            sum_all = self.data_count*len(self.no_outputs_count)
            prob = sum_no_out/sum_all
            score_fp.write(f"sum:{sum_no_out}/{sum_all}={prob}")

    def test_mean_vec(self, model_name, method, epochs_id, ds, output_len, input_size):

        output_fp = open(f"outputs/{self.theta}/{model_name}-{method}-{epochs_id}-output.txt","w")
        score_fp = open(f"outputs/{self.theta}/{model_name}-{method}-{epochs_id}-score.txt","w")

        jsonencoder = json.JSONEncoder(ensure_ascii=False)  

        model_path = f"/nfs/music-5-test/{model_name}/model/{method}/{epochs_id}.ckpt"  # 指定模型保存路径  

        ds.return_path = True
        data_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

        model = MLP(input_size , 8192, output_len).to(device) 
        model.load_state_dict(torch.load(model_path))

        outputs_arr = []
        targets_arr = []

        if method=="s0":
            decoder = decoder_s0
        elif method=="s1":
            decoder = decoder_s1

        f1_decoders = []
        acc_decoders = []
        recall_decoders = []
        no_outputs_count = []
        data_count = 0
        for i in range(len(decoder.encoders)):
            f1_decoders.append([])
            acc_decoders.append([])
            recall_decoders.append([])
            no_outputs_count.append(0)

        for inputs, targets, path in data_loader:  # 在每个批次上执行以下步骤：    
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移到CUDA设备上  
            outputs = model(inputs)  # 前向传播（将输入传递给模型）    
            outputs = torch.sigmoid(outputs)
            # token = torch.topk(outputs[0], 30).indices.tolist()
            output_onehot = (outputs > self.theta).int()
            acc, recall, f1 = F1_score(output_onehot, targets)
            
            output_label = decoder.decode_prob(outputs[0].tolist(),self.theta)
            ori_label = decoder.decode_prob(targets[0].tolist(),self.theta)
            resText = f"acc={acc}\trecall={recall}\tf1={f1}"
            outputs_arr.append(output_onehot[0].tolist())
            targets_arr.append(targets[0].tolist())
            print(path)
            res_json = jsonencoder.encode(
                {
                    "output_label":output_label,
                    "ori_label":ori_label,
                    "score":resText,
                    "path":path[0]
                }
            )
            output_fp.write(f"{res_json}\n")

            #给每个样本打分

            for i in range(len(decoder.encoders)):
                it = decoder.encoders[i]
                outputs_column = output_onehot[:,it.begin_at:it.end_at]
                targets_column = targets[:,it.begin_at:it.end_at]
                output_sum = outputs_column.sum().item()
                target_sum = targets_column.int().sum().item()
                # print(targets_column.shape, targets_column.shape, output_sum, target_sum)
                if target_sum!=0 and output_sum==0:
                    no_outputs_count[i] += 1
                    self.no_outputs_count[i] += 1
                acc, recall, f1 = F1_score(outputs_column.reshape(-1), targets_column.reshape(-1))
                f1_decoders[i].append(f1.item())
                acc_decoders[i].append(acc.item())
                recall_decoders[i].append(recall.item())
            data_count += 1
            self.data_count += 1

        outputs_arr = torch.tensor(outputs_arr)
        targets_arr = torch.tensor(targets_arr)

        for i in range(len(decoder.encoders)):
            it = decoder.encoders[i]
            outputs_column = outputs_arr[:,it.begin_at:it.end_at]
            targets_column = targets_arr[:,it.begin_at:it.end_at]
            
            # acc, recall, f1 = F1_score(outputs_column.reshape(-1), targets_column.reshape(-1))
            
            acc = torch.tensor(acc_decoders[i]).mean().item()
            recall = torch.tensor(recall_decoders[i]).mean().item()
            f1 = torch.tensor(f1_decoders[i]).mean().item()
            
            f1_std = torch.tensor(f1_decoders[i]).std().item()
            acc_std = torch.tensor(acc_decoders[i]).std().item()
            recall_std = torch.tensor(recall_decoders[i]).std().item()
            
            score_fp.write(f"shape:{targets_column.shape}\tacc={acc}\trecall={recall}\tf1={f1}\tf1_std={f1_std}\tacc_std={acc_std}\trecall_std={recall_std}\tno_outputs_count:{no_outputs_count[i]}/{data_count}\n")

    def get_output_len(self, method):
        if method=="s0":
            output_len = 14162
        elif method=="s1":
            output_len = 9691
        return output_len

    def test_mean_vec_mert(self, model_name, method, epochs_id, input_size):
        output_len = self.get_output_len(method)

        dataset_index = f"../datas/{method}_ans_token_test.txt"
        dataset_pt = f"/nfs/music-5-test/{model_name}/encode/"
        ds = VecDatasetSearch(path=dataset_index, output_len=output_len, dir_path=dataset_pt,replace=[("_src.mp3","_mert.pt")], preload=False)  
        self.test_mean_vec(model_name, method, epochs_id, ds, output_len, input_size)


    def test_mean_vec_music2vec(self, model_name, method, epochs_id, input_size):
        output_len = self.get_output_len(method)

        dataset_index = f"../datas/{method}_ans_token_test.txt"
        dataset_pt = f"/nfs/music-5-test/{model_name}/encode/"
        ds = VecDatasetMusic2Vec(path=dataset_index, output_len=output_len ,replace=[("/nfs/datasets-mp3/",dataset_pt),("_src.mp3","_src.pkl")], preload=False)  
        self.test_mean_vec(model_name, method, epochs_id, ds, output_len, input_size)

    def test_mean_vec_jukebox(self, model_name, method, epochs_id, input_size):
        output_len = self.get_output_len(method)

        dataset_index = f"../datas/{method}_ans_token_test.txt"
        dataset_pt = f"/nfs/music-5-test/{model_name}/encode/"
        ds = VecDatasetJukeBox(path=dataset_index, output_len=output_len ,replace=[("/nfs/datasets-mp3/",dataset_pt),("_src.mp3","_src.pkl.jkb")], preload=False)  
        self.test_mean_vec(model_name, method, epochs_id, ds, output_len, input_size)

    def test_mean_vec_encodec(self, model_name, method, epochs_id, input_size):
        output_len = self.get_output_len(method)

        dataset_index = f"../datas/{method}_ans_token_test.txt"
        dataset_pt = f"/nfs/music-5-test/{model_name}/encode/"
        ds = VecDatasetEncodec(path=dataset_index, output_len=output_len,replace=[("/nfs/datasets-mp3/",dataset_pt),("_src.mp3","_src.pkl.mean")], preload=False)  
        self.test_mean_vec(model_name, method, epochs_id, ds, output_len, input_size)

    def process(self):
        self.make_global_score()
        self.test_mean_vec_mert("mert300", "s0", 19980, 1024)
        self.test_mean_vec_mert("mert95", "s0", 19980, 768)
        self.test_mean_vec_music2vec("music2vec", "s0", 19980, 768)
        self.test_mean_vec_jukebox("jukebox", "s0", 19980, 4800)
        self.test_mean_vec_encodec("encodec", "s0", 19980, 19200)
        self.save_global_score("s0")

        self.make_global_score()
        self.test_mean_vec_mert("mert300", "s1", 19980, 1024)
        self.test_mean_vec_mert("mert95", "s1", 19980, 768)
        self.test_mean_vec_music2vec("music2vec", "s1", 19980, 768)
        self.test_mean_vec_jukebox("jukebox", "s1", 19980, 4800)
        self.test_mean_vec_encodec("encodec", "s1", 19980, 19200)
        self.save_global_score("s1")

# ModelScore(0.4).process()
ModelScore(0.5).process()
ModelScore(0.6).process()
ModelScore(0.7).process()
ModelScore(0.8).process()
ModelScore(0.9).process()