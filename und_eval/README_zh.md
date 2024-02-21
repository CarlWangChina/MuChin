# 5种模型测试代码

## 配置环境

``` python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 数据预处理  

``` python
python inference/encodec-inference.py
python inference/jukebox-inference.py
python inference/music2vec-inference.py
python inference/extract_mert_features/extract_mert_features.py 95
python inference/extract_mert_features/extract_mert_features.py 330

python tools/buildTagDict.py
python tools/data_preprocess.py
python tools/encodec_decode.py
python tools/get_data_list.py
python tools/mean_jukebox.py
```

## 模型训练  

`sh train.sh`  

## 模型测试

`python predict/predict.py`  
