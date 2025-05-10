import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Ayarlar
MODEL_PATH = "models/final_model.keras"
IMAGE_DIR = "custom_images"
IMAGE_SIZE = (28, 28)
DEBUG_DIR = "debug_images"

# Eğitim sırasında kullanılan etiket eşleştirmesi
LABEL_MAP = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
    12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
    18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Test amaçlı
def center_crop_26x26(img):
    """28x28 görüntünün ortasındaki 25x25 alanı döndürür."""
    h, w = img.shape[:2]
    start_x = (w - 26) // 2
    start_y = (h - 26) // 2
    return img[start_y:start_y + 26, start_x:start_x + 26]


def preprocess_image(img_path, debug=False):
    """Model tahmini için görüntüyü işler."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"HATA: {img_path} okunamadı.")
        return None

    # Görüntüyü yeniden boyutlandır
    img = cv2.resize(img, IMAGE_SIZE)


    # Normalleştir ve modele uygun şekle getir
    img_norm = img.astype("float32") / 255.0
    img_norm = np.expand_dims(img_norm, axis=-1)
    img_norm = np.expand_dims(img_norm, axis=0)

    # Debug için kaydet
    if debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        debug_path = os.path.join(DEBUG_DIR, os.path.basename(img_path))
        cv2.imwrite(debug_path, img)

    return img_norm, img  # hem işlenmiş hem görselleştirilecek ham hâli

def predict_and_display(model, img_path):
    """Görseli işler, tahmin yapar ve sonucu gösterir."""
    processed, display_img = preprocess_image(img_path, debug=True)
    if processed is None:
        return

    pred = model.predict(processed)[0]
    predicted_idx = np.argmax(pred)
    confidence = pred[predicted_idx]
    predicted_char = LABEL_MAP[predicted_idx]

    # Görsel ile birlikte sonucu göster
    plt.figure(figsize=(12, 4))

    # Orijinal görüntü
    plt.subplot(1, 3, 1)
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(original, cmap='gray')
    plt.title(f"Orijinal: {os.path.basename(img_path)}")

    # İşlenmiş görüntü
    plt.subplot(1, 3, 2)
    plt.imshow(display_img, cmap='gray')
    plt.title("İşlenmiş")

    # Tahmin dağılımı
    plt.subplot(1, 3, 3)
    plt.bar(range(len(pred)), pred, color='skyblue')
    plt.xticks(range(len(pred)), list(LABEL_MAP.values()), rotation=90)
    plt.title(f"Tahmin: {predicted_char} ({confidence:.2f})")

    plt.tight_layout()
    plt.show()

def main():
    # Model kontrolü
    if not os.path.exists(MODEL_PATH):
        print(f"Model bulunamadı: {MODEL_PATH}")
        print("Lütfen önce eğitim dosyasını çalıştırın.")
        return

    model = load_model(MODEL_PATH)

    # Görsel klasör kontrolü
    if not os.path.exists(IMAGE_DIR):
        print(f"Görsel klasörü bulunamadı: {IMAGE_DIR}")
        return

    # Görselleri işle
    for img_file in sorted(os.listdir(IMAGE_DIR)):
        if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(IMAGE_DIR, img_file)
            predict_and_display(model, img_path)

if __name__ == "__main__":
    main()
