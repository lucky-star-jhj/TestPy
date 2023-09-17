import torch  
import torch.nn as nn  
import torchvision.transforms as transforms  
import torchvision.datasets as dsets  
from torch.autograd import Variable   # Variable类在PyTorch中用于封装Tensor，并可以自动计算梯度
import matplotlib.pyplot as plt  
  
# 定义SoftMax函数  
class Softmax(nn.Module):  
    def __init__(self, num_classes):  
        super(Softmax, self).__init__()  
        self.num_classes = num_classes  
        # 会创建一个线性层，其输入特征的数量是28*28（即784，这通常是MNIST数据集的特征数），
        # 输出特征的数量是num_classes。num_classes是你分类的类别数量
        self.linear = nn.Linear(28*28, num_classes)   
  
    def forward(self, x):  
        x = x.view(-1, 28*28)  
        x = self.linear(x)  
        return nn.functional.softmax(x, dim=1)  
  
# 加载MNIST数据集  
train_dataset = dsets.MNIST(root='./data',  
                            train=True,   
                            transform=transforms.ToTensor(),  
                            download=True)  
  
test_dataset = dsets.MNIST(root='./data',  
                           train=False,   
                           transform=transforms.ToTensor())  
  
# 数据加载器  
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,  
                                           batch_size=100,   
                                           shuffle=True)  
  
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  
                                          batch_size=100,   
                                          shuffle=False)  
  
# 实例化Softmax分类器  
classifier = Softmax(num_classes=10)  
  
# 选择训练策略和优化器  
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)  
criterion = nn.CrossEntropyLoss()  
  
# 训练模型  
device = torch.device('cpu')  # 使用GPU
for epoch in range(10):  
    for i, (images, labels) in enumerate(train_loader):  
        images = Variable(images.float()).to(device)
        labels = Variable(labels).to(device)  
        outputs = classifier(images)  
        loss = criterion(outputs, labels)  
        optimizer.zero_grad()  # 梯度清0
        loss.backward()  # 计算梯度
        optimizer.step()  # 优化梯度
        if (i+1) % 100 == 0:  
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'   
                   .format(epoch+1, 10, i+1, len(train_loader), loss.item()))  
  
# 测试模型  
correct = 0  
total = 0  
for images, labels in test_loader:  
    images = Variable(images.float())  
    labels = Variable(labels)  
    outputs = classifier(images)  
    _, predicted = torch.max(outputs.data, 1)  
    total += labels.size(0)  
    correct += (predicted == labels).sum()  
print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))