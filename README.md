# MuChin: A Benchmark for Evaluating Multimodal Language Models on Music Description
## News
- 04/2024 We are excited to announce that our paper on MuChin has been accepted by IJCAI 2024! 
## Introduction
MuChin is the first open-source music description benchmark in Chinese colloquial language. It aims to evaluate the performance of multimodal Large Language Models (LLMs) in understanding and describing music. We introduce the CaiChong Music Annotation Platform (CaiMAP) and the Caichong Music Dataset (CaiMD) to facilitate this evaluation.
![image](https://github.com/CarlWangChina/MuChin/assets/41322045/5dbd4bb4-0923-4304-a275-a33884b8b1d4)
## CaiChong Music Annotation Platform (CaiMAP)
CaiMAP utilizes an innovative multi-person, multi-stage assurance method to ensure the precision of annotations and alignment with popular semantics. This platform recruits both amateurs and professionals for annotation.
## Caichong Music Dataset (CaiMD)
CaiMD is a dataset built using CaiMAP's annotation method. It contains multi-dimensional, high-precision music annotations. We have carefully selected 1,000 high-quality entries from CaiMD to serve as the test set for MuChin.
## Usage
To evaluate a model on MuChin, use the scoring code provided to calculate the model's performance on the test set. We also provide detailed appendices with analysis of the performance of various models on MuChin.
## Analysis
We conducted an analysis on the discrepancies between professional and amateur descriptions. Additionally, we demonstrated the effectiveness of using annotated data for fine-tuning LLMs.
## Data Download
Due to the large size of the relevant audio files, they have been uploaded to the Google Drive at link "[google drive muchin](https://drive.google.com/drive/folders/1LA-wjkZSCppX3WULJK8Z5jT4pzJYEKzV?usp=drive_link)" for downloading by interested researchers. The mapping relationship between the audio file IDs and the annotated content is saved in this "mp3_id_to_meta_info.xlsx" Excel file.
