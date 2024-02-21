import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy

class MLP(nn.Module):  
    def __init__(self, input_size, hidden_size, output_size):  
        super(MLP, self).__init__()  
        self.mlp = nn.Sequential(  
            nn.Linear(input_size, hidden_size),  
            nn.ReLU(),  
            nn.Linear(hidden_size, hidden_size),  
            nn.ReLU(),  
            nn.Linear(hidden_size, hidden_size),  
            nn.ReLU(),  
            nn.Linear(hidden_size, hidden_size),  
            nn.ReLU(),  
            nn.Linear(hidden_size, output_size),  
        )
  
    def forward(self, x):
        x = self.mlp(x)
        return x  


class MLPEmbedding(nn.Module):  
    def __init__(self, input_size, embed_out_size, hidden_size, output_size):  
        super(MLPEmbedding, self).__init__()  
        self.embedding = nn.Embedding(num_embeddings=1024, embedding_dim=embed_out_size)
        self.input_layer = nn.Linear(embed_out_size*input_size, hidden_size)
        self.mlp = nn.Sequential(  
            nn.Linear(hidden_size, hidden_size),  
            nn.ReLU(),  
            nn.Linear(hidden_size, hidden_size),  
            nn.ReLU(),  
            nn.Linear(hidden_size, hidden_size),  
            nn.ReLU(),  
            nn.Linear(hidden_size, hidden_size),  
            nn.ReLU(),  
            nn.Linear(hidden_size, output_size),  
        )
  
    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size,-1)
        x = self.embedding(x).view(batch_size*seq_len,-1)
        x = self.input_layer(x).view(batch_size,seq_len,-1)
        x = x.mean(dim=1)
        x = self.mlp(x)
        return x  

def F1_score(data1: torch.IntTensor, data2: torch.IntTensor):
    """
    data1: prediction
    data2: ground truth
    """
    assert data1.shape == data2.shape
    data1 = data1.view(-1)
    data2 = data2.view(-1)
    acc = torch.sum(data1 == data2) / len(data1)
    if acc == 1:
        return torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)
    precision = (
        torch.sum(data1[data1>0]==data2[data1>0]) /
        (torch.sum(data1>0) + 1e-5)
    )
    recall = (
        torch.sum(data1[data2>0]==data2[data2>0]) /
        (torch.sum(data2>0) + 1e-5)
    )
    score = 2 * precision * recall / (precision + recall + 1e-5)
    return acc, recall, score



# testinput = torch.randint(0,1000,[1,2000,150*8],dtype=int)
# m = MLPEmbedding(150*8,128,1024,8192)
# r = m(testinput)
# print(r.shape)