import torch

def calculate_accuracy_and_recall(model, dataset):
    # 设置模型为评估模式
    model.eval()
    
    total_samples = len(dataset)
    correct_samples = 0
    true_positives = 0
    actual_positives = 0
    
    with torch.no_grad():
        for i in range(total_samples):
            inputs, labels = dataset[i]
            inputs = inputs.unsqueeze(0)  # 在第0维增加一个维度，以符合模型输入要求
            
            outputs = model(inputs)
            predicted_labels = torch.argmax(outputs, dim=1)
            
            if predicted_labels == labels:
                correct_samples += 1
                
            if predicted_labels == 1 and labels == 1:
                true_positives += 1
                
            if labels == 1:
                actual_positives += 1
    
    accuracy = correct_samples / total_samples
    recall = true_positives / actual_positives
    
    return accuracy, recall
