# 5 Types of Model Testing Code

## Configuring the Environment

``` python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Preprocessing  

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

Examples of the data format are provided in `datas`

## Model Training  

`sh train.sh`  

## Model Testing

`python predict/predict.py`  
