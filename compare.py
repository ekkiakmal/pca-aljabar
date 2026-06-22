import cv2
import numpy as np
import joblib

pca = joblib.load("pca_model.pkl")

size = (100, 100)

def get_feature(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise Exception(f"Gambar tidak terbaca: {path}")

    img = cv2.resize(img, size)
    vec = img.flatten().reshape(1, -1)
    return pca.transform(vec)[0]

# FOTO (folder "akhir")
foto_anak = r"D:\aljabar projek\akhir\gallen1.jpeg"
foto_dewasa = r"D:\aljabar projek\akhir\gallen2.jpeg"

f1 = get_feature(foto_anak)
f2 = get_feature(foto_dewasa)

distance = np.linalg.norm(f1 - f2)

print("Distance:", distance)

# threshold awal (bisa dituning)
threshold = 9000

if distance < threshold:
    print("KESIMPULAN: Kemungkinan orang yang sama")
else:
    print("KESIMPULAN: Orang berbeda")