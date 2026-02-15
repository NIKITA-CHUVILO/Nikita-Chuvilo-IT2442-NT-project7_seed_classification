import torch
import torch.nn as nn
import torchvision.models as models

class SeedClassifier(nn.Module):
    """
    Классификатор сортов растений на основе EfficientNet-B0
    """
    def __init__(self, num_classes=12, pretrained=True):
        super(SeedClassifier, self).__init__()
        
        # Загружаем предобученный EfficientNet-B0
        # EfficientNet даёт лучшую точность при меньшем количестве параметров
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Получаем размер признаков от backbone
        # У EfficientNet классификатор находится в backbone.classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Заменяем классификатор на новый
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class SeedClassifierResNet(nn.Module):
    """
    Альтернативная модель на ResNet50 (если EfficientNet не работает)
    """
    def __init__(self, num_classes=12, pretrained=True):
        super(SeedClassifierResNet, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes, device, model_type='efficientnet'):
    """
    Утилита для создания модели
    model_type: 'efficientnet' или 'resnet'
    """
    if model_type == 'efficientnet':
        model = SeedClassifier(num_classes=num_classes)
    else:
        model = SeedClassifierResNet(num_classes=num_classes)
    
    model = model.to(device)
    return model