from torch.utils.data import Dataset,DataLoader
import pandas as pd
import torch
import torch.nn as nn


class TitanicDataset(Dataset):    
    def __init__(self, file_path):        
        self.file_path = file_path
        # 预计算的均值，用于标准化数据
        self.mean = {
            "Pclass": 2.236695,
            "Age": 29.699118,
            "SibSp": 0.512605,
            "Parch": 0.431373,
            "Fare": 34.694514,
            "Sex_female": 0.365546,
            "Sex_male": 0.634454,
            "Embarked_C": 0.182073,
            "Embarked_Q": 0.039216,
            "Embarked_S": 0.775910        }

        # 预计算的标准差，用于标准化数据
        self.std = {
            "Pclass": 0.838250,
            "Age": 14.526497,
            "SibSp": 0.929783,
            "Parch": 0.853289,
            "Fare": 52.918930,
            "Sex_female": 0.481921,
            "Sex_male": 0.481921,
            "Embarked_C": 0.386175,
            "Embarked_Q": 0.194244,
            "Embarked_S": 0.417274        }

        # 加载数据
        self.data = self._load_data()
        # 特征数量
        self.feature_size = len(self.data.columns) - 1    
    def _load_data(self):        
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"]) ##删除不用的列        
        df = df.dropna(subset=["Age"])##删除Age有缺失的行        
        #这里的get_dummies是进行one-hot编码
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)##进行one-hot编码       
        ##进行数据的标准化        
        base_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        for i in range(len(base_features)):
            df[base_features[i]] = (df[base_features[i]] - self.mean[base_features[i]]) / self.std[base_features[i]]
        return df

    def __len__(self):        
        return len(self.data)

    def __getitem__(self, idx):   
        #获取特征，根据索引获取样本的特征
        features = self.data.drop(columns=["Survived"]).iloc[idx].values
        #获取标签，根据索引获取样本的标签
        label = self.data["Survived"].iloc[idx]
        #转化为tensor
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# 定义模型，逻辑损失模型
class LogisticRegressionModel(nn.Module):    
    def __init__(self, input_dim):        
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # nn.Linear也继承自nn.Module，输入为input_dim,输出一个值    
    def forward(self, x):        
        return torch.sigmoid(self.linear(x))  # Logistic Regression 输出概率

# 引入数据集和测试集
train_dataset = TitanicDataset("./titanic/train.csv")
validation_dataset = TitanicDataset("./titanic/test.csv")

# 实例化模型
model = LogisticRegressionModel(train_dataset.feature_size)
model.to("cpu")
model.train()

# 定义优化器，传入模型的参数，并且设置固定的学习率
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 定义训练轮数
epochs = 100

for epoch in range(epochs):
    # 定义正确率
    correct = 0
    step = 0
    # 定义总损失
    total_loss = 0
    for features, labels in DataLoader(train_dataset, batch_size=256, shuffle=True):
        step += 1
        # 将特征和标签转移到cpu
        features = features.to("cpu")
        labels = labels.to("cpu")
        optimizer.zero_grad()
        # 前向传播
        outputs = model(features).squeeze()
        correct += torch.sum(((outputs >= 0.5) == labels))
        # 计算损失
        loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
        # 累加损失
        total_loss += loss.item()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {total_loss/step:.4f}')
    print(f'Training Accuracy: {correct / len(train_dataset)}')