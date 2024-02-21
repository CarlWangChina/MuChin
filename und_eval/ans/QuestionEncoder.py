import os
import json
import torch

# 单个问题的编码器
class QuestionEncoder:
    def __init__(self, data:set[str]=None):
        self.kw_mapper = dict()
        self.token_mapper = dict()  # 反向映射
        self.begin_at = 0
        self.end_at = 0
        if not data is None:
            self.load(data)
    
    def load(self, data:set[str])->None:
        kwid = 0
        self.kw_mapper = dict()
        self.token_mapper = dict()
        for it in data:
            self.kw_mapper[it] = kwid
            self.token_mapper[kwid] = it
            kwid += 1
    
    def loadMapper(self,mapper:dict[str,int])->None:
        self.kw_mapper = dict()
        self.token_mapper = dict()
        for k,v in mapper.items():
            self.kw_mapper[k] = v
            self.token_mapper[v] = k
    
    def encode(self, kw:str)->int:
        return self.kw_mapper[kw] + self.begin_at
    
    def decode(self, token:int)->str:
        return self.token_mapper.get(token - self.begin_at, None)

class EncoderGroup:
    def __init__(self, num:int=11, data:list[set[str]]=None):
        if not data is None:
            num = len(data)
        self.encoders = []
        self.kwsum = 0
        for i in range(num):
            if not data is None:
                self.encoders.append(QuestionEncoder(data=data[i]))
            else:
                self.encoders.append(QuestionEncoder())
        if not data is None:
            self.init()
    
    def init(self):
        kwsum = 0
        for it in self.encoders:
            it.begin_at = kwsum
            kwsum += len(it.kw_mapper)
            it.end_at = kwsum
        self.kwsum = kwsum
    
    def encode(self, kws:list[set[str]]):
        res = []
        for i in range(len(kws)):
            if i < len(self.encoders):
                for kw in kws[i]:
                    token = self.encoders[i].encode(kw)
                    res.append(token)
        return res
    
    def decode(self, tokens:list[int]):
        res = []
        for i in range(len(self.encoders)):
            res.append(set())
        for token in tokens:
            for i in range(len(self.encoders)):
                encoder = self.encoders[i]
                keyword = encoder.decode(token)
                if keyword is not None:
                    res[i].add(keyword)
                    break
        
        for i in range(len(self.encoders)):
            res[i] = list(res[i])
        
        return res


    def decode_prob(self, prob:list[float], theta:float):
        token = []
        for i in range(len(prob)):
            if prob[i]>theta:
                token.append(i)
        # print(prob)
        # print(token)
        outputs = torch.tensor(prob)
        outputs_onehot = (torch.tensor(prob)>theta).int()
        for it in self.encoders:
            if outputs_onehot[it.begin_at:it.end_at].sum()==0:
                # print(outputs[it.begin_at:it.end_at])
                force_id = outputs[it.begin_at:it.end_at].argmax().item() + it.begin_at
                token.append(force_id)
        return self.decode(token)
    
    def decode_prob_topk(self, prob:list[float]):
        token = []
        # print(prob)
        
        for it in self.encoders:
            topk = torch.topk(prob[it.begin_at:it.end_at],3).indices.tolist()
            for t in topk:
                token.append(t+it.begin_at)
        
        # print(token)
        return self.decode(token)

    def to_json(self):
        encoders_data = [encoder.kw_mapper for encoder in self.encoders]
        
        jsonencoder = json.JSONEncoder(ensure_ascii=False)  
        json_str = jsonencoder.encode(encoders_data)
                
        return json_str
    
    def save_to_file(self, filename):
        encoders_data = [encoder.kw_mapper for encoder in self.encoders]

        jsonencoder = json.JSONEncoder(ensure_ascii=False)  
        json_str = jsonencoder.encode(encoders_data)

        with open(filename, 'w') as file:
            file.write(json_str)

    @classmethod
    def from_json(cls, json_data):
        encoders_data = json.loads(json_data)
        
        res = cls()
        res.encoders = []
        res.kwsum = 0
        
        for it in encoders_data:
            enc = QuestionEncoder()
            enc.loadMapper(it)
            res.encoders.append(enc)
        
        res.init()
        
        return res

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'r') as file:
            encoders_data = json.load(file)
            
            res = cls()
            res.encoders = []
            res.kwsum = 0

            for it in encoders_data:
                enc = QuestionEncoder()
                enc.loadMapper(it)
                res.encoders.append(enc)

            res.init()

            return res