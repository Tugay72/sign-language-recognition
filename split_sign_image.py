import cv2
import os

# Görsel yolu
image_path = "custom_images/amer_sign2.png"

# Kayıt klasörü
output_dir = "split_letters"
os.makedirs(output_dir, exist_ok=True)

# Görseli yükle
image = cv2.imread(image_path)
if image is None:
    print("Görsel yüklenemedi!")
    exit()

# Görsel boyutları (satır, sütun)
rows = 4
cols = 6

# Her bir hücrenin boyutlarını hesapla
cell_height = image.shape[0] // rows
cell_width = image.shape[1] // cols

# Harfleri sırayla al (J ve Z yok çünkü veri setinde yok)
letters = [
    'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y'
]

# Görseli böl
index = 0
for row in range(rows):
    for col in range(cols):
        if index >= len(letters):
            break
        x_start = col * cell_width
        y_start = row * cell_height
        cropped = image[y_start:y_start + cell_height, x_start:x_start + cell_width]
        filename = os.path.join(output_dir, f"{letters[index]}.png")
        cv2.imwrite(filename, cropped)
        print(f"{letters[index]} harfi kaydedildi → {filename}")
        index += 1

print("\n✅ Tüm harfler başarıyla bölünüp kaydedildi.")
