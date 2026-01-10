import time
from torch.utils.data import DataLoader, Dataset

# 定义一个简单的数据集
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 创建数据集和 DataLoader
dataset = MyDataset([i for i in range(10000)])
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
print(len(dataloader))

print(f"dataloader: {dataloader}")
# print(iter(dataloader))
dataiter = iter(dataloader)
print(next(dataiter))
print(next(dataiter))
print(next(dataiter))

# print(dataloader[0])
print("__________________________________")

curr_time = time.perf_counter()
# 遍历数据加载器
for idx, batch in enumerate(dataloader):
    time
