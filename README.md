# MuChin
The manuscript titled "MuChin: A Chinese Colloquial Description Benchmark for Evaluating Language Models in the Field of Music," slated for publication at the International Joint Conference on Artificial Intelligence 2024 (IJCAI 2024), presents a data set meticulously arranged, featuring several hundred thousand instances. Currently, a subset of this dataset comprising 1,000 instances has been made publicly available as open-source within this repository. This benchmark is notable not just for its comprehensive coverage of both amateur and professional descriptions but also includes intricate structural metadata such as musical sections and rhyme structures. We cordially invite scholars and practitioners to employ this resource broadly in their research initiatives and to properly reference its use in their academic publications. Due to the considerable size of the related audio files, they will be uploaded to Google Drive for access in a timely manner.

Our dataset has been organized within the `dataset` directory. 

The code for testing five different audio understanding models utilizing a Multilayer Perceptron (MLP) approach is located in the `aud_eval` folder; this code processes input audio and outputs tags along ten auditory perception dimensions.

The code dedicated to assessing semantic similarity has been archived within the `semantic` directory. This code receives two sets of tags and computes a similarity score as output.

Moreover, the code for evaluating lyric generation is contained in the `gen_eval` folder. It accepts inputs in the form of musical sections, abstract lyrical representations denoted by sequences such as 'ccccR', as well as complete lyrics, in order to calculate the corresponding scores.

![image](https://github.com/CarlWangChina/MuChin/assets/41322045/5dbd4bb4-0923-4304-a275-a33884b8b1d4)

