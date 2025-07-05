import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import time
import numpy as np

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据加载和预处理
def load_data(batch_size=64):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 加载训练集和测试集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# LeNet-5 模型实现
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第一个卷积层，输入1通道，输出6通道，卷积核大小5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        # 第二个卷积层，输入6通道，输出16通道，卷积核大小5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # 卷积层 -> 激活函数 -> 池化层
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 展平操作
        x = x.view(-1, 16 * 5 * 5)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 基本的残差块
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入和输出维度不匹配，使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out

# ResNet 模型实现
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 残差层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 全连接层
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 创建ResNet-18模型
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# 简化版ResNet，适用于MNIST
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        
        self.linear = nn.Linear(32 * 7 * 7, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 训练函数
def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Accuracy: {100. * correct / total:.2f}%')
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    return test_loss, test_acc

# 可视化训练过程
def plot_results(train_losses, test_losses, train_accs, test_accs, model_name):
    try:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='训练损失')
        plt.plot(test_losses, label='测试损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title(f'{model_name} - 损失曲线')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='训练准确率')
        plt.plot(test_accs, label='测试准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率 (%)')
        plt.title(f'{model_name} - 准确率曲线')
        plt.legend()
        
        plt.tight_layout()
        
        # 确保文件名有效
        filename = f'{model_name}_results.png'
        print(f"正在保存图片到: {filename}")
        plt.savefig(filename)
        print(f"图片已保存到: {filename}")
        
        # 关闭图形以释放内存
        plt.close()
    except Exception as e:
        print(f"生成或保存图片时出错: {e}")

# 主函数
def main():
    # 超参数设置
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    
    # 加载数据
    train_loader, test_loader = load_data(batch_size)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 模型列表
    models = {
        'LeNet': LeNet(),
        'SimpleResNet': SimpleResNet(),
        'ResNet18': ResNet18()
    }
    
    results = {}
    
    # 训练和评估每个模型
    for model_name, model in models.items():
        print(f"\n开始训练 {model_name}...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch)
            test_loss, test_acc = test(model, test_loader, criterion, device)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        training_time = time.time() - start_time
        print(f"{model_name} 训练完成，耗时: {training_time:.2f} 秒")
        
        # 保存模型
        torch.save(model.state_dict(), f'{model_name}_mnist.pth')
        
        # 可视化结果
        plot_results(train_losses, test_losses, train_accs, test_accs, model_name)
        
        # 存储结果以便比较
        results[model_name] = {
            'final_train_acc': train_accs[-1],
            'final_test_acc': test_accs[-1],
            'training_time': training_time
        }
    
    # 比较不同模型的性能
    print("\n模型性能比较:")
    print("-" * 60)
    print(f"{'模型名称':<15} {'训练准确率':<15} {'测试准确率':<15} {'训练时间(秒)':<15}")
    print("-" * 60)
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['final_train_acc']:<15.2f} {result['final_test_acc']:<15.2f} {result['training_time']:<15.2f}")
    print("-" * 60)

if __name__ == '__main__':
    main()