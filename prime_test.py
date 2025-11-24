import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tabulate import tabulate


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCustom(nn.Module):
    def __init__(self, num_classes=7, dropout_p=0.3):
        super(ResNetCustom, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


transform_func = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),  # вот это протестировать ещё нужно ли
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

def generate_crops(image, n_crops=25):
    """Создаёт последовательность кропов с разной степенью обрезки."""
    w, h = image.size
    crop_ratios = np.linspace(0.0, 0.2, n_crops)
    crops = []

    for r in crop_ratios:
        r_w = r * 0.5  # ширина урезается в 2 раза медленнее
        r_h = r        # высота уменьшается сильнее

        left = int(w * r_w)
        right = int(w * (1 - r_w))
        top = int(h * r_h)
        bottom = int(h * (1 - r_h))

        cropped = image.crop((left, top, right, bottom))
        crops.append(cropped)

    return crops

def transform_crops(crops, transform, device):
    """Применяет трансформации к каждому кропу и возвращает батч тензоров."""
    transformed = []

    for crop in crops:
        img_t = transform(crop).unsqueeze(0).to(device)  # [1,3,H,W]
        transformed.append(img_t)

    return transformed


def predict_from_crops(model, transformed_crops, class_names):
    """Получает предсказания от модели для всех трансформированных кропов
       и усредняет вероятности."""
    preds = []

    with torch.no_grad():
        for img_t in transformed_crops:
            outputs = model(img_t)  # [1, num_classes]
            preds.append(outputs)

    avg_pred = torch.stack(preds).mean(0)  # [1, num_classes]
    probs = torch.softmax(avg_pred, dim=1).cpu().numpy().flatten()

    # — красивый вывод таблицы
    rows = [[name, f"{p:.4f}"] for name, p in zip(class_names, probs)]
    print(tabulate(rows, headers=["Класс", "Вероятность"], tablefmt="github"))

    final_class = class_names[avg_pred.argmax().item()]
    print(f"\nФинальный вывод (усреднение по кропам): {final_class}")

    return final_class, probs

def predict_with_crop_tta(image_path, model, class_names, transform, device, n_crops=25):
    """Основная функция, использующая 3 подфункции."""
    image = Image.open(image_path).convert("RGB")

    # --------------------------
    # 0) Показываем исходное изображение
    # --------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Исходное изображение")
    plt.show()

    # 1: создаём кропы
    crops = generate_crops(image, n_crops=n_crops)

    # 2: трансформируем
    transformed = transform_crops(crops, transform, device)

    # --------------------------
    # ПОКАЗ: последний кроп после трансформаций
    # --------------------------
    last_t = transformed[-1].squeeze(0).cpu()  # [3,H,W]
    last_t = last_t * 0.5 + 0.5  # denormalize
    last_t = last_t.numpy().transpose(1, 2, 0)  # [H,W,3]

    # 3: получаем предсказание
    final_class, probs = predict_from_crops(model, transformed, class_names)

    plt.figure(figsize=(6, 6))
    plt.imshow(last_t)
    plt.axis("off")
    plt.title(f"Предсказание: {final_class}")
    plt.show()

device = torch.device("cpu")
num_classes = 7
model = ResNetCustom(num_classes=num_classes)
model.load_state_dict(torch.load("best_resnet_model_actual.pth", map_location=device))
model.to(device)
model.eval()
print("Модель загружена и готова к тестированию")

url = "https://www.inomarkispb.ru/image/catalog/blog/vidy-sedanov/6521_3.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")
image.save("test_car.jpg")

class_names = ['Convertible', 'Coupe', 'Hatchback', 'Pick-Up', 'SUV', 'Sedan', 'VAN']
predict_with_crop_tta("test_car.jpg", model, class_names, transform_func, device)
