import os
import cv2
import numpy as np
import joblib
from sklearn.decomposition import PCA

dataset_path = r"D:\aljabar projek\FGNET\images"

X = []
size = (100, 100)

for file in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, file)

    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    img = cv2.resize(img, size)
    X.append(img.flatten())

X = np.array(X)

print("Jumlah gambar:", len(X))

# PCA
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

joblib.dump(pca, "pca_model.pkl")

print("Training selesai")