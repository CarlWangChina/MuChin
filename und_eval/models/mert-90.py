import os
import sys
import json
import traceback
import torch
import torchaudio
import librosa
from typing import Optional
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import pyloudnorm as pyln
import torch  
import torch.multiprocessing as mp  

class mert90Processor:
    def __init__(
            self,
            mert_model_name: str = "m-a-p/MERT-v1-330M",
            cache_dir = "/home/feee/cache/",
            mert_feature_rate: int = 75,
            window_size: int = 60,
            device: Optional[str] = None,
    ):
        if device is None:
            if torch.cuda.device_count() > 0:
                self.device: str = "cuda:0"
            elif torch.backends.mps.is_available():
                self.device: str = "mps"
            else:
                self.device: str = "cpu"
        else:
            self.device: str = device
        print("device:",self.device)

        self.mert_model = AutoModel.from_pretrained(
            mert_model_name,trust_remote_code=True,cache_dir=cache_dir,force_download=False
        ).to(self.device)
        
        self.mert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            mert_model_name,cache_dir=cache_dir,force_download=False
        )
        
        self.mert_sampling_rate = self.mert_feature_extractor.sampling_rate
        self.mert_feature_rate = mert_feature_rate
        self.mert_num_channels = 1
        self.window_size = window_size
        self.padding = self.mert_sampling_rate // self.mert_feature_rate
    
    def get_mert_features(self,
                           waveform: torch.Tensor,
                           output_file_name: str):
        """Extract features from audio file.

        Args:
            waveform (torch.Tensor): waveform tensor.  Should be 1D tensor.
            output_file_name (str):  name of the output feature file to write into.
        """

        assert len(waveform.shape) == 1, "waveform should be 1D tensor."

        # Get the feature tensor window by window.
        mert_features = torch.zeros(0)
        window_size_in_samples = self.window_size * self.mert_sampling_rate + self.padding
        window_size_in_features = self.window_size * self.mert_feature_rate

        with torch.no_grad():
            with (gzip.open(output_file_name, "wb") as output_file):
                for i in range(0, len(waveform), window_size_in_samples):
                    waveform_window = waveform[i: i + window_size_in_samples + self.padding]

                    data_size_in_window = window_size_in_features

                    if len(waveform_window) < window_size_in_samples:
                        #按声音长度切窗口
                        data_size_in_window = math.ceil(data_size_in_window*(len(waveform_window)/window_size_in_samples))
                        waveform_window = torch.cat(
                            (
                                waveform_window,
                                torch.zeros(window_size_in_samples - len(waveform_window)),
                            )
                        )

                    feature = self.mert_feature_extractor(
                        waveform_window,
                        sampling_rate=self.mert_sampling_rate,
                        padding=True,
                        return_attention_mask=True,
                        return_tensors="pt",
                    )
                    feature["input_values"] = feature["input_values"].to(self.device)
                    feature["attention_mask"] = feature["attention_mask"].to(self.device)

                    feature = self.mert_model(**feature)
                    assert feature["last_hidden_state"].shape[1] >= window_size_in_features, \
                        "The last_hidden_state should be at least as long as the window size in features."

                    feature = feature["last_hidden_state"][0, :data_size_in_window, :
                              ].to("cpu").type(torch.float32).flatten()
                    output_file.write(feature.numpy().tobytes())

                    del feature
                    
    def process(self, dataArray, sample_rate):
        
        if not dataArray is None:
            #归一化音量到-12
            meter = pyln.Meter(sample_rate) # create BS.1770 meter
            loudness = meter.integrated_loudness(dataArray)
            loudness_normalized_audio = pyln.normalize.loudness(dataArray, loudness, -12.0)
            # loudness_current = meter.integrated_loudness(loudness_normalized_audio
            # print(loudness,"->",loudness_current)

            audio_input_to_mert = librosa.resample(loudness_normalized_audio, orig_sr=sample_rate, target_sr=self.mert_sampling_rate)
            
            #命名成其他名字，处理完再重命名，防止处理过程中中断导致错误
            tmp_out = outFile+".tmp"
            
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
            if os.path.exists(outFile):
                os.remove(outFile)
            
            self.get_mert_features(torch.tensor(audio_input_to_mert), tmp_out)

            os.rename(tmp_out, outFile)
            print(outFile)

mert90Processor()