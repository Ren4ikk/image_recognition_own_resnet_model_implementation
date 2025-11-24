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


# ---------------- A0: Модель ----------------

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


device = torch.device("cpu")
num_classes = 7
model = ResNetCustom(num_classes=num_classes)
model.load_state_dict(torch.load("best_resnet_model_actual.pth", map_location=device))
model.to(device)
model.eval()

print("Модель загружена и готова к тестированию")

# ---------------- A1: Базовая предобработка изображения ----------------

# Здесь только приведение к RGB и первичный resize (если нужно).
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
    # можно убрать, если кропы считаешь по оригиналу
])

# ---------------- A2: Правила предобработки + TTA-кропы ----------------

# A1: предобработка всего изображения
def preprocess_image(image_path):
    """
    A1: Получение изображения и полная предобработка transform.
    Возвращает тензор [C, H, W].
    """
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image)      # здесь уже применяем transform
    return img_t


# A2: формирование кропов ИЗ уже предобработанного тензора
def make_tta_crops(img_tensor, n_crops=1):
    """
    A2: Формирование набора TTA-кропов из предобработанного тензора.
    На входе: тензор [C, H, W] после transform.
    Возвращает список тензоров кропов и PIL‑картинку последнего кропа для визуализации.
    """
    # размеры тензора
    _, h, w = img_tensor.shape

    crop_ratios = np.linspace(0.0, 0.2, n_crops)
    tensors = []
    last_crop_tensor = None

    for r in crop_ratios:
        r_w = int(w * r * 0.05)
        r_h = int(h * r)

        left = r_w
        right = w - r_w
        top = r_h
        bottom = h - r_h

        # кроп по тензору: [C, H, W] -> [C, h_crop, w_crop]
        crop_t = img_tensor[:, top:bottom, left:right]
        last_crop_tensor = crop_t
        tensors.append(crop_t)

    # последний кроп в PIL для plt.imshow
    vis_crop = transforms.ToPILImage()(last_crop_tensor.cpu())

    return tensors, vis_crop


# ---------------- A3: Инференс по кропам и усреднение ----------------

def infer_with_tta(crop_tensors, model, device):
    """
    A3: Инференс модели по каждому кропу и усреднение выходов.
    Возвращает усреднённый тензор логитов [1, num_classes].
    """
    preds = []
    model.eval()
    with torch.no_grad():
        for img_t in crop_tensors:
            img_t = img_t.unsqueeze(0).to(device)  # [1, 3, H, W]
            outputs = model(img_t)
            preds.append(outputs)

    avg_pred = torch.stack(preds).mean(0)  # [1, num_classes]
    return avg_pred


# ---------------- A4: Постобработка, выбор класса, визуализация ----------------

def postprocess_and_visualize(avg_pred, class_names, cropped_image, n_crops=25):
    """
    A4: Расчёт вероятностей, выбор класса, вывод таблицы и визуализация.
    """
    probs = torch.softmax(avg_pred, dim=1).cpu().numpy().flatten()

    rows = []
    for name, p in zip(class_names, probs):
        rows.append([name, f"{p:.4f}"])

    print(tabulate(rows, headers=["Класс", "Вероятность"], tablefmt="github"))

    class_index = avg_pred.argmax().item()
    print(f"Финальное предсказание (усреднённое по {n_crops} кропам): "
          f"{class_names[class_index]}")

    plt.imshow(cropped_image)
    plt.axis("off")
    plt.title(f"Предсказано: {class_names[class_index]}")
    plt.show()

    return class_names[class_index]


# ---------------- Объединяющая функция (A0) ----------------

def predict_with_pipeline(image_path, model, class_names, device, n_crops=25):
    # A1: предобработка до тензора
    prepared_tensor = preprocess_image(image_path)

    # A2: кропы из этого тензора
    crop_tensors, vis_crop = make_tta_crops(prepared_tensor, n_crops=n_crops)

    # A3: инференс
    avg_pred = infer_with_tta(crop_tensors, model, device)

    # A4: постобработка
    final_class = postprocess_and_visualize(avg_pred, class_names, vis_crop, n_crops=n_crops)
    return final_class


# ---------------- Пример использования ----------------

url = "https://www.wheelka.ru/i/uploads/images/CighAE8Uma4.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")
image.save("test_car.jpg")

class_names = ['Convertible', 'Coupe', 'Hatchback', 'Pick-Up', 'SUV', 'Sedan', 'VAN']

predict_with_pipeline("test_car.jpg", model, class_names, device)
