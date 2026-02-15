import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os
from tqdm import tqdm
import argparse
import json
from pathlib import Path

from model import get_model

def setup_dataloaders(data_dir, batch_size=32, img_size=224):
    """
    Подготовка данных с расширенной аугментацией
    """
    # Трансформации для тренировочных данных (сильная аугментация)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(45),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])
    
    # Трансформации для валидационных данных (без аугментации)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Загружаем весь датасет
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    
    # Разделяем на train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Для валидации используем transform без аугментации
    val_dataset.dataset.transform = val_transform
    
    # Создаем загрузчики
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2, pin_memory=True)
    
    # Сохраняем имена классов для отчёта
    class_names = full_dataset.classes
    
    return train_loader, val_loader, class_names

def train_epoch(model, loader, criterion, optimizer, device):
    """Одна эпоха обучения"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Валидация модели"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_confusion_matrix(labels, preds, class_names, save_path):
    """Построение матрицы ошибок"""
    cm = confusion_matrix(labels, preds)
    
    # Нормализуем для лучшей визуализации
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Абсолютные значения
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix (Absolute)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Нормализованные значения
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_history(history, save_path):
    """Построение графиков обучения"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # График потерь
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Training and Validation Loss')
    axes[0].grid(True)
    
    # График точности
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def calculate_per_class_accuracy(labels, preds, class_names):
    """Расчёт точности по каждому классу"""
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        class_mask = np.array(labels) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(preds)[class_mask] == i)
            per_class_acc[class_name] = float(class_acc)
        else:
            per_class_acc[class_name] = 0.0
    return per_class_acc

def main():
    parser = argparse.ArgumentParser(description='Train Seed Classification Model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset (should contain class folders)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model_type', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet'],
                       help='Model architecture')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    args = parser.parse_args()
    
    # Создаем директорию для результатов
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Загружаем данные
    print("\nLoading data...")
    train_loader, val_loader, class_names = setup_dataloaders(
        args.data_dir, args.batch_size
    )
    print(f"Found {len(class_names)} classes")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Создаем модель
    print(f"\nCreating {args.model_type} model...")
    model = get_model(
        num_classes=len(class_names), 
        device=device,
        model_type=args.model_type
    )
    
    # Выводим количество параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1, verbose=True
    )
    
    # История обучения
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Цикл обучения
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Сохраняем историю
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Обновляем learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"✓ Saved new best model with val_acc: {val_acc:.4f}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best model from epoch {best_epoch} with val_acc: {best_val_acc:.4f}")
    
    # Финальная оценка
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Загружаем лучшую модель
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Получаем финальные предсказания
    _, _, final_preds, final_labels = validate(
        model, val_loader, criterion, device
    )
    
    # Итоговые метрики
    final_acc = accuracy_score(final_labels, final_preds)
    final_f1_macro = f1_score(final_labels, final_preds, average='macro')
    final_f1_weighted = f1_score(final_labels, final_preds, average='weighted')
    
    print(f"\nFinal Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Test Accuracy: {final_acc:.4f}")
    print(f"F1-Score (macro): {final_f1_macro:.4f}")
    print(f"F1-Score (weighted): {final_f1_weighted:.4f}")
    
    # Подробный отчёт по классам
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(final_labels, final_preds, 
                                 target_names=class_names, digits=4)
    print(report)
    
    # Сохраняем отчёт
    with open(os.path.join(args.save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Сохраняем confusion matrix
    plot_confusion_matrix(final_labels, final_preds, class_names,
                         os.path.join(args.save_dir, 'confusion_matrix.png'))
    
    # Сохраняем графики обучения
    plot_training_history(history, 
                         os.path.join(args.save_dir, 'training_history.png'))
    
    # Сохраняем метрики в JSON
    per_class_acc = calculate_per_class_accuracy(final_labels, final_preds, class_names)
    
    metrics = {
        'best_val_accuracy': best_val_acc,
        'final_test_accuracy': final_acc,
        'final_f1_macro': final_f1_macro,
        'final_f1_weighted': final_f1_weighted,
        'per_class_accuracy': per_class_acc,
        'training_history': {
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_loss': history['val_loss'],
            'val_acc': history['val_acc']
        }
    }
    
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ All results saved to {args.save_dir}")
    print("\nProject completed successfully!")
    
    # Выводим топ-3 лучших и худших классов
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 3 best classified classes:")
    for class_name, acc in sorted_classes[:3]:
        print(f"  {class_name}: {acc:.4f}")
    
    print("\nTop 3 worst classified classes:")
    for class_name, acc in sorted_classes[-3:]:
        print(f"  {class_name}: {acc:.4f}")

if __name__ == '__main__':
    main()