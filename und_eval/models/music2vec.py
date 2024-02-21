from transformers import Wav2Vec2Processor, Data2VecAudioModel
import torch
from torch import nn
from torchaudio.transforms import Resample


class music2vecEncoder(nn.Module):
    def __init__(self,
                 device: str,
                 maxLength :int=7680000):
        super(music2vecEncoder, self).__init__()
        self.device = device

        self.model = Data2VecAudioModel.from_pretrained("m-a-p/music2vec-v1",cache_dir="/home/feee/cache/",force_download=False).to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h",cache_dir="/home/feee/cache/",force_download=False)
        self.maxLength = maxLength
        self.inputSamplingRate = 16000
    
    @torch.no_grad()
    def encode(self, data:torch.Tensor, sampling_rate:int):
        assert data.dim()==1
        if sampling_rate!=self.inputSamplingRate:
            rs = Resample(sampling_rate, self.inputSamplingRate)
            data = rs(data)
        
        if data.shape[0]>self.maxLength:
            data = data[:self.maxLength]

        inputs = self.processor(data, sampling_rate=self.inputSamplingRate, return_tensors="pt")

        outputs = self.model(input_values=inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda"), output_hidden_states=True)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        return all_layer_hidden_states[12]
