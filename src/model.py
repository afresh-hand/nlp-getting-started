import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, random_split

class myDataset(Dataset):
    def __init__(self, x_data, y_data) -> None:
        self.xData = x_data
        self.yData = y_data
        self.length = x_data.size(0)
        
        
    def __getitem__(self, index):
        return self.xData[index], self.yData[index]
    
    def __len__(self):
        return self.length
    

class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()
    
    def forward(self, din):
        return