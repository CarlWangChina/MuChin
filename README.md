# MuChin
The IJCAI2024 paper “MuChin: A Chinese Colloquial Description Benchmark for Evaluating Language Models in the Field of Music” contains an open source selection of high-quality 1k datasets: In addition to the professional/amateur descriptions primarily discussed in the paper, it also includes structural information such as musical sections and rhymes. We welcome everyone to make extensive use of it and cite it. The audio package is quite large and it will be uploaded to Google Drive later and then linked to our GitHub.

Our dataset has been organized within the `dataset` directory. 

The code for testing five different audio understanding models utilizing a Multilayer Perceptron (MLP) approach is located in the `aud_eval` folder; this code processes input audio and outputs tags along ten auditory perception dimensions.

The code dedicated to assessing semantic similarity has been archived within the `semantic` directory. This code receives two sets of tags and computes a similarity score as output.

Moreover, the code for evaluating lyric generation is contained in the `gen_eval` folder. It accepts inputs in the form of musical sections, abstract lyrical representations denoted by sequences such as 'ccccR', as well as complete lyrics, in order to calculate the corresponding scores.

![image](https://github.com/CarlWangChina/MuChin/assets/41322045/5dbd4bb4-0923-4304-a275-a33884b8b1d4)

