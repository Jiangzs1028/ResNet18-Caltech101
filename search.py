import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import os
from sklearn.model_selection import train_test_split
import itertools
import copy

# 定义搜索空间
batch_sizes = [256, 512]
lrs = [0.01, 0.001]
step_sizes = [10, 5]
gammas = [0.1, 0.5]
num_epochs = 40

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

    return Subset(dataset, train_idx), Subset(dataset, test_idx)

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
                num_epochs=25, device='cuda'):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

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

        # 每个epoch的训练指标
        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = running_corrects.double() / total_samples

        # 验证集准确率
        epoch_val_acc = test_model(model, val_loader, device)

        # 学习率调度器更新
        scheduler.step()

        # 输出日志
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {epoch_train_loss:.4f} | '
              f'Train Acc: {epoch_train_acc:.4f} | '
              f'Val Acc: {epoch_val_acc:.4f}')

        # 记录最优模型
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return best_acc

# 网格搜索主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = load_caltech101()


    # 所有组合
    search_space = list(itertools.product(batch_sizes, lrs, step_sizes, gammas))
    results = []

    for batch_size, lr, step_size, gamma in search_space:
        print(f"\n🔍 当前组合: batch_size={batch_size}, lr={lr}, step_size={step_size}, gamma={gamma}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

        model = initialize_model()
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        params = [
            {'params': model.fc.parameters(), 'lr': lr},         # 输出层较大学习率
            {'params': [p for n, p in model.named_parameters() if "fc" not in n], 'lr': lr / 10}  # 其余层较小学习率
        ]
        optimizer = optim.SGD(params, momentum=0.9)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        best_acc = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                               num_epochs=num_epochs, device=device)

        results.append({
            'batch_size': batch_size,
            'lr': lr,
            'step_size': step_size,
            'gamma': gamma,
            'accuracy': best_acc
        })

        print(f"✅ 完成 | 最佳准确率: {best_acc:.4f}")

    # 打印所有结果
    print("\n🎯 所有组合结果:")
    for r in results:
        print(f"[batch_size={r['batch_size']} lr={r['lr']} step={r['step_size']} gamma={r['gamma']}] → acc: {r['accuracy']:.4f}")

    # 选出最优组合
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n🏆 最优组合: {best}")

if __name__ == '__main__':
    main()
