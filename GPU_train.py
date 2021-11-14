import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 确认我们的电脑支持CUDA，然后显示CUDA信息：
print(device)
# 返回gpu数量
GPU_num = torch.cuda.device_count()
# 返回gpu名字，设备索引默认从0开始
GPU_name = torch.cuda.get_device_name(0)
# 返回当前设备索引
GPU_index = torch.cuda.current_device()
print(GPU_num, GPU_index, GPU_name)


# 读入数据集
import dataset
trainloader, testloader, classes = dataset.dataset()

# 加载网络结构
import resnet18
net = resnet18.ResNet(resnet18.ResidualBlock, [3,3,3], 10)
net.to(device) # 转换成CUDA张量

import os
# if os.path.exists("./model/mymodel.pkl"):
#     net.load_state_dict(torch.load("./model/mymodel.pkl"))
#     print("model loaded...")

# 定义损失函数和优化器
import torch.optim as optim
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 训练网络
import time # 记录训练时间
start = time.time()



for epoch in range(20):  # 多批次循环

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) # inputs, targets 和 images 也要转换
        # 梯度置0
        optimizer.zero_grad()

        # 正向传播，反向传播，优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印状态信息
        running_loss += loss.item()
        # print(running_loss)
    print(f'epoch{epoch} loss: {running_loss}')
    running_loss = 0.0
        # if i % 100 == 99:    # 每1000批次打印一次， 在 gpu 上训练请调大此参数
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 100))
        #     running_loss = 0.0


print('Finished Training')
print('程序的运行时间：%.2f' % (time.time() - start), "s")
# 快速保存我们训练好的模型：
torch.save(net.state_dict(), "./model/mymodel.pkl")
print('model saved')
