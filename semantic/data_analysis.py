from FlagEmbedding import FlagModel
import os
from tqdm import tqdm
import pandas as pd
import json
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from sklearn.cluster import KMeans
import pickle
from config import gt
import json


class MuChindata_Analyzer:
    '''
    Cleaning and merging the Prompt part of Muer annotation data, and comparing the difference between amateur and
    professional annotators.
    '''
    def __init__(self , filename, model_dir):
        # Setting use_fp16 to True speeds up computation with a slight performance degradation
        self.sentence_model = FlagModel(model_dir,query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：", use_fp16=True) 
        self.data = filename


    def data_warping(self, directory, group = "专业组", return_labels=True):
        def merge(filename, return_labels=False):
            '''
            Merge the labeled data and save it
            parameters:
                mode: mode, mode=1 means counting s0, mode=2 means counting s1, mode=3 means counting the result of merging s0 and s1.
                return_labels: True means the last returned labels, False means the last returned description.
            return_value:
                Store the extracted structure as a dictionary, return the dictionary.
            '''
            res = {"歌名":filename.split('/')[-1].replace('.txt','')}
            index = 0
            questions = [ "" for _ in range(20) ]
            answers = [ "" for _ in range(20) ]
            with open(filename, 'r', encoding='utf-8') as f:  
                lines = f.readlines()  
                passage_line = ''
                if return_labels:
                    for line in lines:  
                        # Determine if it is an answer line
                        if line.startswith('Q') or line.startswith('\n') or line.startswith('@'):
                            continue
                        else:
                            line = line.replace('\t', '').replace('\n', '')                
                            if line.startswith('-q'):
                                questions[index] = line[5:]
                                index += 1
                            elif line.startswith('la') or line.startswith('oa'):
                                answers[index-1] = line[5:]
                            passage_line+=line
                    for i in range(index):
                        res[questions[i]] = answers[i].split(',')
                else:
                    for line in lines:  
                        #   
                        if line.startswith('Q') or line.startswith('\n') or line.startswith('@'):
                            continue
                        else:
                            line = line.replace('\t', '').replace('\n', '')
                            if line.startswith('TA'):
                                questions[index] = '这首歌带给你的感受'
                                answers[index] = line[4:]
                                index += 1
                            elif line.startswith('-q'):
                                questions[index] = line[5:]
                                index += 1
                            elif line.startswith('la') or line.startswith('oa'):
                                answers[index-1] = line[5:]
                    for i in range(index):
                        res[questions[i]] = answers[i]
                return res
        '''
        Process all labeling information for professional/non-professional groups, organize and export to excel file
        '''
        files = [directory + '/' + f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        data = []
        for i in tqdm(range(len(files)), desc=f"Processing messages from {group}"):
            data.append(merge(files[i], return_labels = return_labels))
        df = pd.DataFrame(data)
        excel_path = f'./data/{group}_labels_{return_labels}.xlsx'
        df.to_excel(excel_path, index=False)


    def solve_labels(self, cmp_kinds, self_kinds_amateur):
        # When cmp_kinds are the same, compare the self_kinds labeling data of professional and amateur annotators
        for cmp_kind in cmp_kinds:
            with open(f'data/divided_data_professional/{cmp_kind}.json') as f:
                cluster = json.load(f)
            for i in range(len(self_kinds_amateur)):
                self.compare_discrepancy(cluster, cmp_kind, self_kinds_amateur[i], True)                


    def compare_discrepancy(self, clusters, cmp_type, self_type, has_labels = False):
        '''
        Comparing the differences between professional and amateur labelers
        '''
        print(f"Beginning to compare cognitive differences in {self_type} labelers when {cmp_type} is the same")
        res = {}
        data = pd.read_excel(self.data, sheet_name=None)
        data_b = data['Sheet1']
        data_a = data['Sheet2']
        data_a.set_index('歌名', inplace=True)
        data_b.set_index('歌名', inplace=True)
        for center_name, points in clusters.items():
            print(f"Counting {center_name}...")
            tmp = []
            for i in tqdm(range(len(points))): 
                for song_name in data_a.index: 
                    try:
                        words = data_a.loc[song_name, cmp_type]
                        point = points[i]
                        if point in words and song_name in data_b.index:
                            amateur_feel = [data_b.loc[song_name, self_type]]
                            profess_feel = [data_a.loc[song_name, self_type]]
                            embeddings_1 = self.sentence_model.encode(amateur_feel)
                            embeddings_2 = self.sentence_model.encode(profess_feel)
                            similarity = embeddings_1 @ embeddings_2.T 
                            tmp.append(str(similarity[0][0]))                      
                    except:
                        continue
            res[center_name] = tmp
        with open(f'exp/edu/{cmp_type}_cmp_{self_type}.json', "w", encoding="utf-8") as ff:
            json.dump(res, ff, ensure_ascii=False, indent=4)


    def compmodel_eval(self):
        '''
        Evaluating the Effectiveness of Understanding Models
        '''
        def get_data_from_txt(filename):
            # Read the content of the file
            with open(filename, "r") as file:
                content = file.read()
            dict_strs = content.strip().split('\n')
            # Initialize an empty list to store dictionaries
            data = []
            # Iterate over the dictionary strings and parse them
            for dict_str in dict_strs:
                try:
                    # Parse the string into a dictionary
                    parsed_dict = eval(dict_str)
                    data.append(parsed_dict)
                except Exception as e:
                    print(f"Error parsing dictionary: {e}")
            return data
        

        def compare_labels(output, ori):
            """
            Compare the semantic similarity between 'output_label' and 'ori_label' for each dictionary in data.
            """
            similarity_list = []
            
            for out, ori_ in zip(output, ori):
                similarity = self.cal_word_similarity(out, ori_)
                similarity_list.append(similarity)
            # print(similarity_list)
            return similarity_list

        for file in os.listdir('data/cpmodel_eval'):
            if 'output' in file:
                fn  = 'data/cpmodel_eval' + '/' + file
                fn_split = file.split('.')[-2]
                if os.path.exists(f'exp/cpeval_bf/{fn_split}.npy'):
                    continue
                print(f'Processing{fn}....')
                data = get_data_from_txt(fn)
                # Extract 'output_label' and 'ori_label' from the data
                output_labels = [d['output_label'] for d in data]
                ori_labels = [d['ori_label'] for d in data]
                # Calculate the semantic similarity for each pair of labels
                similarity_results = np.array(compare_labels(output_labels, ori_labels))
                np.save(f'exp/cpeval_bf/{fn_split}.npy',similarity_results)


    def cal_word_similarity(self, output, ori):
        '''
        Calculate similarity between words using bruce-force
        '''
        res = []
        for i in tqdm(range(len(output))):
            tmp = []
            for out_d in output[i]:
                for ori_d in ori[i]:
                    embeddings_1 = self.sentence_model.encode([out_d])
                    embeddings_2 = self.sentence_model.encode([ori_d])
                    similarity = embeddings_1 @ embeddings_2.T 
                    tmp.append(similarity[0][0])     
            tmp_array = np.array(tmp)

            mean = np.mean(tmp_array)
            std_dev = np.std(tmp_array)
            res.append([mean, std_dev]) 
        
        return res

    
if __name__ == '__main__':
    Muer = MuChindata_Analyzer('data/labels-free.xlsx')
    Muer.compmodel_eval()
    