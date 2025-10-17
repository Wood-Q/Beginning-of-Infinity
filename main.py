import torch

# a=torch.zeros(2,3)
# print(a)

# b=torch.randn(2,3)
# print(b)

# c=torch.randn(2,3)
# print(c)

# print(b+c)

# print(b.t())

# print(c.shape)

# print(torch.cuda.is_available())

# 创建一个需要梯度的张量
# tensor_requires_grad = torch.tensor([1.0], requires_grad=True)
# print(tensor_requires_grad)

# # 进行一些操作
# tensor_result = tensor_requires_grad * 2
# print(tensor_result)  # 输出结果

# # 计算梯度
# tensor_result.backward()
# print(tensor_requires_grad.grad)  # 输出梯度

from torch import nn
from torch import optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1=nn.Linear(2,2)
        self.fc2=nn.Linear(2,1)
    
    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=self.fc2(x)
        return x

model=SimpleNN()

criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr=0.01)

# 4. 假设我们有训练数据 X 和 Y
X = torch.randn(10, 2)  # 10 个样本，2 个特征
Y = torch.randn(10, 1)  # 10 个目标值

for epoch in range(100):
    optimizer.zero_grad()  # 清除梯度
    outputs = model(X)  # 前向传播
    loss = criterion(outputs, Y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')