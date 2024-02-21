import os
import json
import dataset

def getFile(path,train_set,test_set):
    count = 0
    train_fp = open(train_set,"w")
    test_fp = open(test_set,"w")
    with open(path) as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            if line!="":
                data = json.loads(line)
                try:
                    dataset.findFileByName(os.path.basename(data["path"].replace("_src.mp3","_mert.pt")),
                                           ["/nfs/music-5-test/mert300/encode/0/","/nfs/music-5-test/mert300/encode/1/"])
                    if os.path.exists(data["path"]):
                        count += 1
                        if count>=4500:
                            test_fp.write(f"{line}\n")
                        else:
                            train_fp.write(f"{line}\n")
                except Exception:
                    pass
            line = fp.readline()

getFile("../datas/s0_ans_token.txt", "../datas/s0_ans_token_train.txt", "../datas/s0_ans_token_test.txt")
getFile("../datas/s1_ans_token.txt", "../datas/s1_ans_token_train.txt", "../datas/s1_ans_token_test.txt")
