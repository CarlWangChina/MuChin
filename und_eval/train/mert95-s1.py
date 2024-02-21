import sys
sys.path.append("./")

import torch  
from models.mlp import MLP,F1_score
from dataset import VecDatasetReplace,VecDatasetSearch
from torch.nn import BCEWithLogitsLoss  
from tqdm import tqdm  
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/train.yaml')

num_epochs = cfg.model.mert95_s1.num_epochs  # 设置训练轮数  
save_path = cfg.model.mert95_s1.save_path  # 指定模型保存路径  
device = cfg.model.mert95_s1.device
dataset_index = cfg.model.mert95_s1.dataset_index
loss_path = cfg.model.mert95_s1.loss_path
output_len = cfg.model.mert95_s1.output_len
dataset_pt = cfg.model.mert95_s1.dataset_pt

ds = VecDatasetSearch(path=dataset_index, output_len=output_len, dir_path=dataset_pt,replace=cfg.model.mert95_s1.dataset)  
data_loader = torch.utils.data.DataLoader(ds, batch_size=cfg.model.mert95_s1.batch_size, shuffle=True)
  
model = MLP(768, 8192, output_len).to(device)
  
criterion = BCEWithLogitsLoss(pos_weight=ds.weight).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # 创建优化器实例（这里以随机梯度下降为例）  

loss_file = open(loss_path, "w")

for epoch in range(num_epochs):    
    with tqdm(total=len(data_loader), ncols=160) as pbar:  # 创建一个进度条  
        for inputs, targets in data_loader:  # 在每个批次上执行以下步骤：    
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移到CUDA设备上  
            optimizer.zero_grad()  # 重置梯度为零    
            outputs = model(inputs)  # 前向传播（将输入传递给模型）    
            loss = criterion(outputs, targets)  # 计算损失（将模型的输出与目标进行比较）    
            loss.backward()  # 反向传播（根据损失计算梯度）    
            optimizer.step()  # 优化步骤（更新权重）    
            pbar.update()  # 更新进度条  
            acc, recall, f1 = F1_score((outputs > 0.51).int(), targets)
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item()} acc:{acc} recall:{recall} f1:{f1}")  # 在进度条中显示当前批次的损失  
            loss_file.write(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item()} acc:{acc} recall:{recall} f1:{f1}\n")  # 将损失写入文件
            
        # 保存模型  
        if epoch%20==0:
            torch.save(model.state_dict(), f"{save_path}/{epoch}.ckpt")