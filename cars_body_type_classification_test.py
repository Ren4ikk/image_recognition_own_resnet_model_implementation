import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from io import BytesIO


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

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),  # вот это протестировать ещё нужно ли
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

def predict_with_crop_tta(image_path, model, class_names, device, n_crops=25):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    preds = []

    crop_ratios = np.linspace(0.0, 0.0, n_crops)

    for r in crop_ratios:
        # по ширине меняем в 2 раза медленнее
        r_w = r * 0.5  # ширина урезается в 2 раза медленнее
        r_h = r  # высота как раньше

        left = int(w * r_w)
        right = int(w * (1 - r_w))

        top = int(h * r_h)
        bottom = int(h * (1 - r_h))

        cropped = image.crop((left, top, right, bottom))
        img_t = transform(cropped).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            preds.append(outputs)

    cropped.save("52.jpg")
    cropped_image = Image.open("52.jpg").convert("RGB")
    avg_pred = torch.stack(preds).mean(0)
    from tabulate import tabulate
    probs = torch.softmax(avg_pred, dim=1).cpu().numpy().flatten()  # если avg_pred shape [1, C]

    rows = []
    for name, p in zip(class_names, probs):
        rows.append([name, f"{p:.4f}"])

    print(tabulate(rows, headers=["Класс", "Вероятность"], tablefmt="github"))

    class_index = avg_pred.argmax().item()
    print(f"Финальное предсказание (усреднённое по {n_crops} кропам): {class_names[class_index]}")

    plt.imshow(cropped_image)
    plt.axis("off")
    plt.title(f"Предсказано: {class_names[class_index]}")
    plt.show()

    return class_names[class_index]

url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcToQtjS6XHgWmhBOQls7fux0veoG1LFuF-eXw&s"
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")

image.save("test_car.jpg")

class_names = ['Convertible', 'Coupe', 'Hatchback', 'Pick-Up', 'SUV', 'Sedan', 'VAN']
predict_with_crop_tta("test_car.jpg", model, class_names, device)
