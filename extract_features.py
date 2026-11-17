#Крок 2
import os
import numpy as np
from PIL import Image
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def extract_features(data_dir="data", save_path="features/features.npy"):
    os.makedirs("features", exist_ok=True)

    print("Завантаження моделі MobileNetV2...")
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.eval()

    preprocess = weights.transforms()
    image_paths = []
    features_list = []

    print("Обробка зображень та витягування векторів...")
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                path = os.path.join(root, file)
                image_paths.append(path)

                try:
                    img = Image.open(path).convert("RGB")
                    tensor = preprocess(img).unsqueeze(0)

                    with torch.no_grad():
                        feat = model.features(tensor)
                        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                        feat = torch.flatten(feat, 1)

                    feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                    features_list.append(feat.numpy().squeeze())
                except Exception as e:
                    print(f"Помилка файлу {path}: {e}")

    features_array = np.array(features_list)

    np.save(save_path, {"paths": image_paths, "features": features_array})
    print(f"Успішно Збережено у {save_path}, Оброблено зображень: {len(image_paths)}")


if __name__ == "__main__":
    extract_features()

