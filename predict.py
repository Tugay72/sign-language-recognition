import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Ayarlar
MODEL_PATH = "models/final_model.keras"
IMAGE_DIR = "custom_images"
IMAGE_SIZE = (28, 28)

# Eğitim sırasında kullanılan etiket eşleştirmesi
LABEL_MAP = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
    12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
    18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

def preprocess_image(img_path):
    """Görseli model için hazırlar"""
    # Görüntüyü oku
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Uyarı: {img_path} okunamadı!")
        return None
    
    # Görüntüyü yeniden boyutlandır
    img = cv2.resize(img, IMAGE_SIZE)
    
    # Görüntüyü tersine çevir (siyah arka plan, beyaz harf)
    img = cv2.bitwise_not(img)
    
    # Gürültüyü azalt
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Eşikleme uygula
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Görüntüyü normalize et
    img = img.astype("float32") / 255.0
    
    # Debug için görüntüyü kaydet
    debug_path = os.path.join("debug_images", os.path.basename(img_path))
    os.makedirs("debug_images", exist_ok=True)
    cv2.imwrite(debug_path, (img * 255).astype(np.uint8))
    
    # Model için gerekli şekle dönüştür
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    return img

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
                predicted_idx = np.argmax(pred)
                confidence = pred[0][predicted_idx]
                predicted_char = LABEL_MAP[predicted_idx]
                
                # Sonucu göster
                print(f"{img_file} → Tahmin: {predicted_char} (Güven: {confidence:.2f})")
                
                # Görselleştirme
                plt.figure(figsize=(12, 4))
                
                # Orijinal görüntü
                plt.subplot(1, 3, 1)
                original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                plt.imshow(original, cmap='gray')
                plt.title(f"Orijinal: {img_file}")
                
                # İşlenmiş görüntü
                plt.subplot(1, 3, 2)
                plt.imshow(processed_img.reshape(IMAGE_SIZE), cmap='gray')
                plt.title("İşlenmiş")
                
                # Tahmin dağılımı
                plt.subplot(1, 3, 3)
                plt.bar(range(len(pred[0])), pred[0])
                plt.title(f"Tahmin: {predicted_char}")
                
                plt.tight_layout()
                plt.show()

if __name__ == "__main__":
    main()
