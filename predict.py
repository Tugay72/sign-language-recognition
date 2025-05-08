# predict.py
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Ayarlar
MODEL_PATH = "models/final_model.keras"
IMAGE_DIR = "custom_images"
IMAGE_SIZE = (28, 28)

def preprocess_image(img_path):
    """Görseli model için hazırlar"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Uyarı: {img_path} okunamadı!")
        return None
    
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]  # Beyaz arka plan için
    return img.reshape(1, *IMAGE_SIZE, 1) / 255.0

def main():
    # Modeli yükle
    try:
        model = load_model(MODEL_PATH)
    except:
        print("Hata: Model bulunamadı! Önce train_model.py'yi çalıştırın.")
        return

    # Klasördeki görselleri işle
    for img_file in os.listdir(IMAGE_DIR):
        if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(IMAGE_DIR, img_file)
            processed_img = preprocess_image(img_path)
            
            if processed_img is not None:
                # Tahmin yap
                pred = model.predict(processed_img)
                predicted_char = chr(np.argmax(pred) + 65)
                
                # Sonucu göster
                print(f"{img_file} → Tahmin: {predicted_char}")
                
                # Görselleştirme (opsiyonel)
                plt.imshow(processed_img.reshape(IMAGE_SIZE), cmap='gray')
                plt.title(f"{img_file} → {predicted_char}")
                plt.show()

if __name__ == "__main__":
    main()

