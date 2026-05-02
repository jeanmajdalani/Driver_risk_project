import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =========================
# 1. PARAMÈTRES
# =========================
DATASET_PATH = r"driver_risk_project/data_raw/yawn_eye"   # change ce chemin selon ton PC
IMG_SIZE = 224
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Classes du dataset
ALL_CLASSES = ["Closed", "Open", "yawn", "no_yawn"]

# Encodage labels
EYE_CLASSES = {
    "Open": 0,
    "Closed": 1
}

YAWN_CLASSES = {
    "no_yawn": 0,
    "yawn": 1
}

# =========================
# 2. VÉRIFICATION + COMPTAGE
# =========================
print("=== Vérification du dataset ===\n")

image_counts = {}
bad_files = []

for cls in ALL_CLASSES:
    class_path = os.path.join(DATASET_PATH, cls)

    if not os.path.exists(class_path):
        print(f"[WARNING] Dossier introuvable : {class_path}")
        continue

    count = 0
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)

        # vérifier extension
        if not file.lower().endswith(VALID_EXTENSIONS):
            bad_files.append(file_path)
            continue

        # vérifier si image lisible
        img = cv2.imread(file_path)
        if img is None:
            bad_files.append(file_path)
            continue

        count += 1

    image_counts[cls] = count

print("Nombre d'images par classe :")
for cls, count in image_counts.items():
    print(f"{cls}: {count}")

print(f"\nNombre de fichiers invalides/corrompus : {len(bad_files)}")

# =========================
# 3. AFFICHER QUELQUES EXEMPLES
# =========================
print("\n=== Affichage de quelques images ===")

plt.figure(figsize=(10, 8))

for i, cls in enumerate(ALL_CLASSES):
    class_path = os.path.join(DATASET_PATH, cls)
    files = [f for f in os.listdir(class_path) if f.lower().endswith(VALID_EXTENSIONS)]

    if len(files) == 0:
        continue

    sample_file = random.choice(files)
    img_path = os.path.join(class_path, sample_file)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.tight_layout()
plt.show()

# =========================
# 4. FONCTION DE CHARGEMENT
# =========================
def load_dataset(dataset_path, class_map, img_size=224):
    X = []
    y = []

    for cls, label in class_map.items():
        class_path = os.path.join(dataset_path, cls)

        if not os.path.exists(class_path):
            print(f"[WARNING] Dossier manquant : {class_path}")
            continue

        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)

            if not file.lower().endswith(VALID_EXTENSIONS):
                continue

            img = cv2.imread(file_path)
            if img is None:
                continue

            # Resize
            img = cv2.resize(img, (img_size, img_size))

            # Convertir BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalisation [0, 1]
            img = img / 255.0

            X.append(img)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y

# =========================
# 5. DATASET YEUX
# =========================
print("\n=== Préparation dataset YEUX ===")

X_eye, y_eye = load_dataset(DATASET_PATH, EYE_CLASSES, IMG_SIZE)

print("Shape X_eye :", X_eye.shape)
print("Shape y_eye :", y_eye.shape)
print("Min pixel X_eye :", X_eye.min())
print("Max pixel X_eye :", X_eye.max())

# split train / val / test
X_train_eye, X_temp_eye, y_train_eye, y_temp_eye = train_test_split(
    X_eye, y_eye, test_size=0.30, random_state=42, stratify=y_eye
)

X_val_eye, X_test_eye, y_val_eye, y_test_eye = train_test_split(
    X_temp_eye, y_temp_eye, test_size=0.50, random_state=42, stratify=y_temp_eye
)

print("\nYEUX - Split terminé")
print("Train :", X_train_eye.shape, y_train_eye.shape)
print("Validation :", X_val_eye.shape, y_val_eye.shape)
print("Test :", X_test_eye.shape, y_test_eye.shape)

# =========================
# 6. DATASET BÂILLEMENT
# =========================
print("\n=== Préparation dataset BÂILLEMENT ===")

X_yawn, y_yawn = load_dataset(DATASET_PATH, YAWN_CLASSES, IMG_SIZE)

print("Shape X_yawn :", X_yawn.shape)
print("Shape y_yawn :", y_yawn.shape)
print("Min pixel X_yawn :", X_yawn.min())
print("Max pixel X_yawn :", X_yawn.max())

# split train / val / test
X_train_yawn, X_temp_yawn, y_train_yawn, y_temp_yawn = train_test_split(
    X_yawn, y_yawn, test_size=0.30, random_state=42, stratify=y_yawn
)

X_val_yawn, X_test_yawn, y_val_yawn, y_test_yawn = train_test_split(
    X_temp_yawn, y_temp_yawn, test_size=0.50, random_state=42, stratify=y_temp_yawn
)

print("\nBÂILLEMENT - Split terminé")
print("Train :", X_train_yawn.shape, y_train_yawn.shape)
print("Validation :", X_val_yawn.shape, y_val_yawn.shape)
print("Test :", X_test_yawn.shape, y_test_yawn.shape)

# =========================
# 7. SAUVEGARDE OPTIONNELLE
# =========================
save_choice = False   # Mets True si tu veux sauvegarder en .npy

if save_choice:
    os.makedirs("driver_risk_project/data_processed", exist_ok=True)

    # Yeux
    np.save("driver_risk_project/data_processed/X_train_eye.npy", X_train_eye)
    np.save("driver_risk_project/data_processed/y_train_eye.npy", y_train_eye)
    np.save("driver_risk_project/data_processed/X_val_eye.npy", X_val_eye)
    np.save("driver_risk_project/data_processed/y_val_eye.npy", y_val_eye)
    np.save("driver_risk_project/data_processed/X_test_eye.npy", X_test_eye)
    np.save("driver_risk_project/data_processed/y_test_eye.npy", y_test_eye)

    # Bâillement
    np.save("driver_risk_project/data_processed/X_train_yawn.npy", X_train_yawn)
    np.save("driver_risk_project/data_processed/y_train_yawn.npy", y_train_yawn)
    np.save("driver_risk_project/data_processed/X_val_yawn.npy", X_val_yawn)
    np.save("driver_risk_project/data_processed/y_val_yawn.npy", y_val_yawn)
    np.save("driver_risk_project/data_processed/X_test_yawn.npy", X_test_yawn)
    np.save("driver_risk_project/data_processed/y_test_yawn.npy", y_test_yawn)

    print("\nDatasets sauvegardés dans data_processed/")

print("\n=== Preprocessing terminé avec succès ===")