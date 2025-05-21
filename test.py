import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import os
import copy

model_path = './model/best_model0.92.pth'
# 测试集预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载 Caltech101 数据集
def load_caltech101_test(root_path='./data'):
    img_dir = os.path.join(root_path, '101_ObjectCategories')
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"数据集路径不存在：{img_dir}")

    dataset = ImageFolder(root=img_dir, transform=test_transform,
                         is_valid_file=lambda x: x.endswith(('.jpg', '.png')) and '__MACOSX' not in x)

    classes = [d.name for d in os.scandir(img_dir) if d.is_dir() and d.name != 'BACKGROUND_Google']
    classes.sort()

    _, test_idx = [], []
    for class_idx, _ in enumerate(classes):
        targets = [i for i, (_, idx) in enumerate(dataset.samples) if idx == class_idx]
        _, class_test = train_test_split(targets, test_size=0.2, random_state=42)
        test_idx.extend(class_test)

    test_dataset = copy.deepcopy(Subset(dataset, test_idx))
    test_dataset.dataset.transform = test_transform
    return test_dataset

# 初始化模型
def initialize_model(num_classes=101):
    model = torchvision.models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model

# 测试模型
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    test_dataset = load_caltech101_test()
    test_loader = DataLoader(test_dataset, batch_size=256, num_workers=8, pin_memory=True)

    # 初始化模型并加载权重
    model = initialize_model(num_classes=101)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # 测试模型
    accuracy = test_model(model, test_loader, device)
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
