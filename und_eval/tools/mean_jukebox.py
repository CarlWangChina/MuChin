import os  
import pickle  
import torch  
import multiprocessing  
from concurrent.futures import ProcessPoolExecutor

# 定义处理单个文件的函数  
def process_file(file_path):  
    data = torch.tensor(torch.load(file_path)[1])  # 加载数据  
    mean = data.mean(dim=0, keepdim=True)  # 计算均值  
    print(mean.shape)
    with open(file_path+".jkb", 'wb') as f:  
        torch.save(mean, f)  # 写回结果  
  
# 定义主函数  
def main():  
    # 指定目录路径  
    p=ProcessPoolExecutor()
    dir_path = '/nfs/music-5-test/jukebox/encode/'  # 修改为你的目录路径  
      
    # 使用os.walk()递归遍历目录  
    for dirpath, dirnames, filenames in os.walk(dir_path):  
        for filename in filenames:  
            file_path = os.path.join(dirpath, filename)  
            if file_path.endswith('.pkl'):  # 仅处理.pkl文件  
                p.submit(process_file, file_path)  
  
if __name__ == '__main__':  
    main()