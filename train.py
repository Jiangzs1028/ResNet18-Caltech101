import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.model_selection import train_test_split
import copy
import datetime
    
# 参数配置
batch_size = 256
lr = 0.01
gamma = 0.5
step_size = 10
num_epochs = 40
use_pretrain=True


# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_caltech101(root_path='./data'):
    img_dir = os.path.join(root_path, '101_ObjectCategories')
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"数据集路径不存在：{img_dir}")

    dataset = ImageFolder(root=img_dir, transform=train_transform,
                         is_valid_file=lambda x: x.endswith(('.jpg', '.png')) and '__MACOSX' not in x)

    classes = [d.name for d in os.scandir(img_dir) if d.is_dir() and d.name != 'BACKGROUND_Google']
    classes.sort()

    train_idx, test_idx = [], []
    for class_idx, _ in enumerate(classes):
        targets = [i for i, (_, idx) in enumerate(dataset.samples) if idx == class_idx]
        class_train, class_test = train_test_split(targets, test_size=0.2, random_state=42)
        train_idx.extend(class_train)
        test_idx.extend(class_test)

    # 应用不同的transform到验证集
    test_dataset = copy.deepcopy(Subset(dataset, test_idx))
    test_dataset.dataset.transform = test_transform
    
    return Subset(dataset, train_idx), test_dataset

def initialize_model(num_classes=101, use_pretrained=True):
    model = torchvision.models.resnet18(pretrained=use_pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def test_model(model, dataloader, device='cuda'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
               num_epochs=25, device='cuda', log_dir=None):
    # 创建TensorBoard writer
    if log_dir is None:
        log_dir = f"runs/caltech101_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # 训练阶段
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
        
        # 计算训练指标
        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = running_corrects.double() / total_samples
        
        # 验证阶段
        epoch_val_acc = test_model(model, val_loader, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
        
        # 打印日志
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {epoch_train_loss:.4f} | '
              f'Train Acc: {epoch_train_acc:.4f} | '
              f'Val Acc: {epoch_val_acc:.4f}')
        
        # 保存最佳模型
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    # 训练结束
    writer.close()
    print(f'Best Val Acc: {best_acc:.4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

def main():
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    train_dataset, val_dataset = load_caltech101()


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    # 初始化模型
    model = initialize_model(use_pretrained=use_pretrain)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 分层学习率设置
    params = [
        {'params': model.fc.parameters(), 'lr': lr},  # 输出层较大学习率
        {'params': [p for n, p in model.named_parameters() if "fc" not in n], 'lr': lr/10}  # 其余层较小学习率
    ]
    optimizer = optim.SGD(params, momentum=0.9)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # 训练参数
    
    # 训练模型
    trained_model = train_model(model, train_loader, val_loader, criterion, 
                               optimizer, scheduler, num_epochs=num_epochs, 
                               device=device)
    
    # 保存模型
    torch.save(trained_model.state_dict(), 'caltech101_resnet18.pth')
    print("Training completed and model saved.")

if __name__ == '__main__':
    main()