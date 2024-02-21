import json

def load_translate_dict(path):
    translate_dict = dict()
    with open(path) as fp:
        data = json.load(fp)
        for k,v in data.items():
            for word in v:
                translate_dict[word] = k
    return translate_dict

translate_dict1 = dict()
translate_dict2 = dict()

translate_dict1["快慢感受"] = load_translate_dict("../datas/divided_data_amateur/快慢感受.json")
translate_dict1["情绪感受（歌词）"] = load_translate_dict("../datas/divided_data_amateur/情绪感受（歌词）.json")
translate_dict1["文化与地域"] = load_translate_dict("../datas/divided_data_amateur/文化与地域.json")
translate_dict1["歌唱和人声"] = load_translate_dict("../datas/divided_data_amateur/歌唱和人声.json")
translate_dict1["歌曲用途"] = load_translate_dict("../datas/divided_data_amateur/歌曲用途.json")
# translate_dict1["歌词和主题内容"] = load_translate_dict("../datas/divided_data_amateur/歌词和主题内容.json")
translate_dict1["特色感受"] = load_translate_dict("../datas/divided_data_amateur/特色感受.json")
translate_dict1["表现力感受（歌手）"] = load_translate_dict("../datas/divided_data_amateur/表现力感受（歌手）.json")
translate_dict1["适用人群"] = load_translate_dict("../datas/divided_data_amateur/适用人群.json")
translate_dict1["配器与音效"] = load_translate_dict("../datas/divided_data_amateur/配器与音效.json")
translate_dict1["音质效果"] = dict()

translate_dict2["情绪感受（歌词和旋律）"] = load_translate_dict("../datas/divided_data_professional/情绪感受（歌词和旋律）.json")
translate_dict2["文化与地域"] = load_translate_dict("../datas/divided_data_professional/文化与地域.json")
translate_dict2["歌唱和人声"] = load_translate_dict("../datas/divided_data_professional/歌唱和人声.json")
translate_dict2["歌曲用途"] = load_translate_dict("../datas/divided_data_professional/歌曲用途.json")
# translate_dict2["歌词和主题内容"] = load_translate_dict("../datas/divided_data_professional/歌词和主题内容.json")
translate_dict2["表现力感受（歌手和伴奏）"] = load_translate_dict("../datas/divided_data_professional/表现力感受（歌手和伴奏）.json")
translate_dict2["适用人群"] = load_translate_dict("../datas/divided_data_professional/适用人群.json")
translate_dict2["速度与节奏"] = load_translate_dict("../datas/divided_data_professional/速度与节奏.json")
translate_dict2["配器与音效"] = load_translate_dict("../datas/divided_data_professional/配器与音效.json")
translate_dict2["风格或流派"] = load_translate_dict("../datas/divided_data_professional/风格或流派.json")
translate_dict2["音质效果"] = dict()

def translate(inpath,outpath,translate_dict):
    ofp = open(outpath,"w")
    with open(inpath) as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            data = json.loads(line)
            if isinstance(data, dict):
                res = dict()
                for k,v in data.items():
                    if isinstance(v, list):
                        dimen_as = list()
                        for ans in v:
                            if ans["dim_name"] in translate_dict:
                                dict_current = translate_dict[ans["dim_name"]]
                                label_as = set()
                                for label in ans["label_as"]:
                                    if label in dict_current:
                                        target = dict_current[label]
                                        # print(label,"->",target)
                                        label_as.add(target)
                                    else:
                                        label_as.add(label)
                                dimen_as.append({"dim_name":ans["dim_name"], "label_as":list(label_as)})
                            # else:
                            #     print(ans["dim_name"])
                        # print(len(dimen_as))
                        res[k] = dimen_as
                    else:
                        res[k] = v
                encoder = json.JSONEncoder(ensure_ascii=False)  
                json_str = encoder.encode(res)
                ofp.write(f"{json_str}\n")
            line = fp.readline()

# {"dim_name": "表现力感受（歌手和伴奏）", "label_as": ["伤感", "忧郁的歌声", "催人泪下"]},
# {"dim_name": "情绪感受（歌词和旋律）", "label_as": ["感性的", "感伤的”", "遗憾", "崩溃"]},
# {"dim_name": "歌词和主题内容", "label_as": ["流行情歌", "悲伤的歌", "现代爱情"]},
# {"dim_name": "配器与音效", "label_as": ["简单的钢琴键和弦", "稳定的鼓节奏", "弹拨小提琴旋律"]},
# {"dim_name": "风格或流派", "label_as": ["创作型歌手/说唱", "创作型歌手/流行音乐", "简单的流行风格鼓乐"]},
# {"dim_name": "速度与节奏", "label_as": ["节奏适中的歌曲", "稳定的鼓点节奏", "重复的伴奏旋律"]},
# {"dim_name": "歌唱和人声", "label_as": ["声音独白", "男灵魂歌声", "男高音歌唱", "深情男歌手"]},
# {"dim_name": "音质效果", "label_as": ["清晰"]},
# {"dim_name": "歌曲用途", "label_as": ["酒吧", "KTV", "失恋的时候"]},
# {"dim_name": "文化与地域", "label_as": ["流行歌曲", "中国小提琴", "现代"]},
# {"dim_name": "适用人群", "label_as": ["恋爱群体", "90后", "00后"]}

translate("../datas/s0_ans.jsonl","../datas/s0_ans_example.jsonl",translate_dict1)
translate("../datas/s1_ans.jsonl","../datas/s1_ans_example.jsonl",translate_dict2)
