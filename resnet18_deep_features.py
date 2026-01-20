import os
import torch
import pandas as pd
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# ================= CONFIG =================
BASE_DIR = Path(__file__).parent.resolve()

SPEC_DIR = BASE_DIR / "Extracted_Features" / "Spectrograms"
OUTPUT_DIR = BASE_DIR / "Deep_Features"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_DIM = 512   # ResNet-18 final feature size
# =========================================


# ---------- Image Transform (ResNet-18) ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # ResNet input
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- Load Pretrained ResNet-18 ----------
resnet = models.resnet18(pretrained=True)

# Remove classification head → feature extractor
resnet.fc = torch.nn.Identity()

resnet.to(DEVICE)
resnet.eval()


# ---------- Feature Extraction Function ----------
def extract_features_from_folder(input_root, output_csv):
    rows = []

    for class_folder in sorted(os.listdir(input_root)):
        class_path = input_root / class_folder
        if not class_path.is_dir():
            continue

        # Class label from folder name: C0_xxx, C1_xxx ...
        label = int(class_folder.split("_")[0][1])

        for img_name in tqdm(os.listdir(class_path), desc=class_folder):
            if not img_name.endswith(".png"):
                continue

            pid = img_name.replace(".png", "")
            img_path = class_path / img_name

            try:
                img = Image.open(img_path).convert("RGB")
                img = transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    feats = resnet(img)   # (1, 512)

                feats = feats.squeeze().cpu().numpy()

                row = {
                    "patient_id": pid,
                    "label": label
                }

                for i, v in enumerate(feats):
                    row[f"resnet_feat_{i+1}"] = v

                rows.append(row)

            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"\n✅ Saved: {output_csv}")
    print("Class distribution:")
    print(df["label"].value_counts())


# ---------- RUN ----------
if __name__ == "__main__":

    print("\n🔹 Extracting ResNet-18 features from SPEECH Mel spectrograms...")
    extract_features_from_folder(
        SPEC_DIR / "speech",
        OUTPUT_DIR / "resnet18_speech_features.csv"
    )

    print("\n🔹 Extracting ResNet-18 features from EGG Mel spectrograms...")
    extract_features_from_folder(
        SPEC_DIR / "egg",
        OUTPUT_DIR / "resnet18_egg_features.csv"
    )
