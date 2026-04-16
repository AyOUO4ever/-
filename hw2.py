import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, densenet121, ResNet18_Weights, DenseNet121_Weights
from torch.utils.data import DataLoader, Subset
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 配置参数（CPU 自适应）
DATA_ROOT = './data'
MAX_TRAIN_TIME = 7200               # 2 小时
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_CPU = (DEVICE.type == 'cpu')
PRINT_FREQ = 20

# CPU 环境下自动调整参数，保证 2 小时内能完成足够轮次
if USE_CPU:
    BATCH_SIZE = 16                 # CPU 时小批量
    NUM_WORKERS = 0                 # 避免多进程开销
    SUBSET_RATIO = 0.2              # 只用 20% 训练数据，仍可对比趋势
    MODELS = ['resnet18', 'densenet121']  # ResNet18 比 ResNeXt 轻量很多
else:
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    SUBSET_RATIO = 1.0
    MODELS = ['resnet18', 'densenet121']  # 您也可以改回 resnext50_32x4d

MODES = ['scratch', 'finetune']
print(f"Running on: {DEVICE} | Batch size: {BATCH_SIZE} | Data subset: {SUBSET_RATIO*100:.0f}%")

# 数据准备（子集可选）
def get_data_loaders():
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,
                                                 download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False,
                                           download=True, transform=transform_test)

    # 取子集加速（仅 CPU 模式下启用）
    if SUBSET_RATIO < 1.0:
        num_samples = int(len(full_trainset) * SUBSET_RATIO)
        indices = np.random.choice(len(full_trainset), num_samples, replace=False)
        trainset = Subset(full_trainset, indices)
    else:
        trainset = full_trainset

    pin_memory = not USE_CPU
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=pin_memory)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin_memory)
    return trainloader, testloader

# 模型构建（兼容新版 weights 参数）
def build_model(model_name, mode='scratch', num_classes=10):
    if model_name == 'resnet18':
        if mode == 'finetune':
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            for param in model.parameters():
                param.requires_grad = False
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            model = resnet18(weights=None, num_classes=num_classes)
    elif model_name == 'densenet121':
        if mode == 'finetune':
            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            for param in model.parameters():
                param.requires_grad = False
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            model = densenet121(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(DEVICE)

# 训练与评估（带时间控制）
def train_one_epoch(model, loader, criterion, optimizer, epoch, mode_name):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f'{mode_name} Epoch {epoch}', leave=False)
    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if i % PRINT_FREQ == 0:
            pbar.set_postfix({'Loss': f'{running_loss/(i+1):.3f}',
                              'Acc': f'{100.*correct/total:.2f}%'})

    return running_loss / len(loader), 100. * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / len(loader), 100. * correct / total

def train_with_time_limit(model, trainloader, testloader, mode_name, max_time):
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'epoch_times': []}
    epoch = 1

    while True:
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, epoch, mode_name)
        test_loss, test_acc = evaluate(model, testloader, criterion)
        scheduler.step(test_loss)

        epoch_time = time.time() - epoch_start
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)

        total_elapsed = time.time() - start_time
        remaining = max_time - total_elapsed
        print(f'{mode_name} Epoch {epoch} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | Time: {epoch_time:.1f}s | '
              f'Total: {total_elapsed/60:.1f}min | Remaining: {remaining/60:.1f}min')

        if total_elapsed >= max_time:
            print(f'[Info] Reached time limit ({max_time}s). Stopping training.')
            break
        epoch += 1

    return history

# 主流程
def main():
    trainloader, testloader = get_data_loaders()
    all_histories = {}
    results_summary = {}

    for model_name in MODELS:
        for mode in MODES:
            mode_name = f'{model_name}_{mode}'
            print(f'\n{"="*60}\nTraining {mode_name}\n{"="*60}')
            model = build_model(model_name, mode=mode)
            history = train_with_time_limit(model, trainloader, testloader,
                                            mode_name, MAX_TRAIN_TIME)
            all_histories[mode_name] = history
            final_acc = history['test_acc'][-1]
            results_summary[mode_name] = final_acc
            torch.save(model.state_dict(), f'{mode_name}_final.pth')
            print(f'Finished {mode_name} | Final Test Acc: {final_acc:.2f}%')

    print('\n' + '='*60)
    print('Summary of Final Test Accuracy (%):')
    for name, acc in results_summary.items():
        print(f'{name:25s}: {acc:.2f}%')
    print('='*60)

    plot_results(all_histories)

def plot_results(histories):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10.colors
    line_styles = {'scratch': '-', 'finetune': '--'}

    for ax, metric in zip(axes[0], ['loss', 'acc']):
        for i, (name, hist) in enumerate(histories.items()):
            mode = 'scratch' if 'scratch' in name else 'finetune'
            style = line_styles[mode]
            color = colors[i % len(colors)]
            epochs = range(1, len(hist[f'train_{metric}'])+1)
            ax.plot(epochs, hist[f'train_{metric}'], linestyle=style, color=color, label=f'{name} train')
            ax.plot(epochs, hist[f'test_{metric}'], linestyle=style, color=color, alpha=0.7, label=f'{name} test')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Training and Test {metric.capitalize()}')
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, alpha=0.3)

    for ax, metric in zip(axes[1], ['loss', 'acc']):
        for i, (name, hist) in enumerate(histories.items()):
            mode = 'scratch' if 'scratch' in name else 'finetune'
            style = line_styles[mode]
            color = colors[i % len(colors)]
            epochs = range(1, len(hist[f'test_{metric}'])+1)
            ax.plot(epochs, hist[f'test_{metric}'], linestyle=style, color=color, label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'Test {metric.capitalize()}')
        ax.set_title(f'Test {metric.capitalize()} Comparison')
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    main()