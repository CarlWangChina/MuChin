import os
import json
from ans.QuestionEncoder import EncoderGroup

def load_file(index_path,data_num):
    tagset = []
    alltagset = set()
    for i in range(data_num):
        tagset.append(set())
    with open(index_path) as fp:
        line = fp.readline()
        count = 0
        while line:
            data = json.loads(line)
            
            for i in range(len(data["dimen_as"])):
                
                for n in data["dimen_as"][i]['label_as']:
                    tagset[i].add(n)
                    alltagset.add(n)
            
            # print(i,len(data["dimen_as"]))
            
            line = fp.readline()
    return tagset
        
def encode_file(index_path,out_path,encoder):
    ofp = open(out_path, "w")
    with open(index_path) as fp:
        line = fp.readline()
        count = 0
        while line:
            data = json.loads(line)
            
            kws = []
            for i in range(len(data["dimen_as"])):
                kws.append(set())
                for n in data["dimen_as"][i]['label_as']:
                    kws[i].add(n)
            
            token = encoder.encode(kws)
            rev = encoder.decode(token)
            
            jsonencoder = json.JSONEncoder(ensure_ascii=False)  
            json_str = jsonencoder.encode({"path":data["audio_lp"],"tag":token})
            ofp.write(f"{json_str}\n")
            
            line = fp.readline()

if __name__ == "__main__":
    enc = EncoderGroup(data=load_file("../datas/s0_ans_example.jsonl",10))
    enc.save_to_file("../datas/s0_ans_dict.txt")
    encode_file("../datas/s0_ans_example.jsonl","../datas/s0_ans_token.txt",enc)
    print("s0 dict len:", enc.kwsum)
    
    enc = EncoderGroup(data=load_file("../datas/s1_ans_example.jsonl",10))
    enc.save_to_file("../datas/s1_ans_dict.txt")
    encode_file("../datas/s1_ans_example.jsonl","../datas/s1_ans_token.txt",enc)
    print("s1 dict len:", enc.kwsum)
