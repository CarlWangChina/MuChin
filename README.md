# MuChin: A Benchmark for Evaluating Multimodal Language Models on Music Description

The 1000 audio WAV files for this dataset, along with the corresponding text annotations for professional descriptions, amateur descriptions, musical segment structure, rhymes, and more, have been uploaded to Hugging Face. Feel free to download them.

https://huggingface.co/datasets/karl-wang/MuChin1k/tree/main

## News
- We have open-sourced an expanded version with 6066 songs. Please refer to this link: https://github.com/CarlWangChina/MuChin-V2-6066
- 04/2024 We are excited to announce that our paper on MuChin has been accepted by IJCAI 2024! Paper: https://www.ijcai.org/proceedings/2024/0860.pdf

## Introduction
The rapidly evolving multimodal Large Language Models (LLMs) urgently require new benchmarks to uniformly evaluate their performance on understanding and textually describing music. However, existing music description datasets are unable to serve as benchmarks due to semantic gaps between Music Information Retrieval (MIR) algorithms and human understanding, discrepancies between professionals and the public, and low precision of annotations.
To address this need, we present MuChin, the first open-source music description benchmark in Chinese colloquial language. It is designed to evaluate the performance of multimodal LLMs in understanding and describing music.
![image](https://github.com/Duoluoluos/MuChin/blob/Dispersion/pic/overview.png)
## The CaiChong Music Annotation Platform (CaiMAP)
We established the CaiChong Music Annotation Platform (CaiMAP) that employs an innovative multi-person, multi-stage assurance method. We recruited both amateurs and professionals to ensure the precision of annotations and alignment with popular semantics.
![image](https://github.com/Duoluoluos/MuChin/blob/Dispersion/pic/annopipe.png)

## MuChin Dataset
Currently, a subset of the MuChin dataset comprising 1,000 instances has been made publicly available as open-source within this repository. The dataset features comprehensive coverage of both amateur and professional descriptions, as well as intricate structural metadata such as musical sections and rhyme structures.

We invite scholars and researchers to employ this resource broadly in their research initiatives. Proper reference to its use in academic publications is appreciated.

Scholars and researchers are requested to obtain relevant song audio from legal channels and use it for academic purposes only. Song audio cannot be used for commercial model training without the authorization of the copyright holder.
Audio download link: https://pan.baidu.com/s/1D4xGQhYUwWbpaHyAS71dfw?pwd=1234 
Extract password: 1234

## Organization
* Our dataset has been organized within the `dataset` directory. 

* The code for testing five different audio understanding models utilizing a Multilayer Perceptron (MLP) approach is located in the `aud_eval` folder; this code processes input audio and outputs tags along ten auditory perception dimensions.

* The code dedicated to assessing semantic similarity has been archived within the `semantic` directory. This code receives two sets of tags and computes a similarity score as output.
## Citation
If you use the MuChin benchmark or dataset in your research, please cite our paper:
```
@inproceedings{wang2024muchin,
  title     = {MuChin: A Chinese Colloquial Description Benchmark for Evaluating Language Models in the Field of Music},
  author    = {Wang, Zihao and Li, Shuyu and Zhang, Tao and Wang, Qi and Yu, Pengfei and Luo, Jinyang and Liu, Yan and Xi, Ming and Zhang, Kejun},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence},
  pages     = {7771--7779},
  year      = {2024},
  month     = {8},
}


