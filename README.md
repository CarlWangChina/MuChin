# MuChin: A Benchmark for Evaluating Multimodal Language Models on Music Description
[[Paper Link]](https://arxiv.org/abs/2402.09871)  [Demo Page]
## News
- 04/2024 We are excited to announce that our paper on MuChin has been accepted by IJCAI 2024!
## Introduction
The rapidly evolving multimodal Large Language Models (LLMs) urgently require new benchmarks to uniformly evaluate their performance on understanding and textually describing music. However, existing music description datasets are unable to serve as benchmarks due to semantic gaps between Music Information Retrieval (MIR) algorithms and human understanding, discrepancies between professionals and the public, and low precision of annotations.
To address this need, we present MuChin, the first open-source music description benchmark in Chinese colloquial language. It is designed to evaluate the performance of multimodal LLMs in understanding and describing music.
![image](https://github.com/CarlWangChina/MuChin/assets/41322045/5dbd4bb4-0923-4304-a275-a33884b8b1d4)
## The CaiChong Music Annotation Platform (CaiMAP)
We established the CaiChong Music Annotation Platform (CaiMAP) that employs an innovative multi-person, multi-stage assurance method. We recruited both amateurs and professionals to ensure the precision of annotations and alignment with popular semantics.
## The Caichong Music Dataset (CaiMD)
Using the CaiMAP method, we built the Caichong Music Dataset (CaiMD), a dataset with multi-dimensional, high-precision music annotations. We carefully selected 1,000 high-quality entries from CaiMD to serve as the test set for MuChin.
## MuChin Dataset
Currently, a subset of the MuChin dataset comprising 1,000 instances has been made publicly available as open-source within this repository. The dataset features comprehensive coverage of both amateur and professional descriptions, as well as intricate structural metadata such as musical sections and rhyme structures.
## Usage
We invite scholars and practitioners to employ this resource broadly in their research initiatives. Proper reference to its use in academic publications is appreciated.
Due to the large size of the relevant audio files, they have been uploaded to the Google Drive at link "[google drive muchin](https://drive.google.com/drive/folders/1LA-wjkZSCppX3WULJK8Z5jT4pzJYEKzV?usp=drive_link)" . The mapping relationship between the audio file IDs and the annotated content is saved in the "mp3_id_to_meta_info.xlsx" Excel file.
## Organization
The dataset has been organized within the `dataset` directory. The code for testing audio understanding models is located in the `aud_eval` folder. The code for assessing semantic similarity is archived within the `semantic` directory. The code for evaluating lyric generation is contained in the `gen_eval` folder.
## Citation
If you use the MuChin benchmark or dataset in your research, please cite our paper:
> @article{wang2024muchin,
>  title={MuChin: A Chinese Colloquial Description Benchmark for Evaluating Language Models in the Field of Music},
>  author={Wang, Zihao and Li, Shuyu and Zhang, Tao and Wang, Qi and Yu, Pengfei and Luo, Jinyang and Liu, Yan and Xi, Ming and Zhang, Kejun},
>  journal={arXiv preprint arXiv:2402.09871},
>  year={2024}
> }
