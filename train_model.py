import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Sabitler
DATA_DIR = 'data'
MODEL_DIR = 'models'
IMAGE_SIZE = (28, 28)
CLASSES = 24
EPOCHS = 50
BATCH_SIZE = 64

# Klasörleri oluştur
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_preprocess_data():
    """Verileri yükler ve ön işlemden geçirir."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'sign_mnist_train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'sign_mnist_test.csv'))

    # J (9) ve Z (25) harflerini çıkar
    train_df = train_df[~train_df['label'].isin([9, 25])].copy()
    test_df = test_df[~test_df['label'].isin([9, 25])].copy()

    # Etiketleri yeniden numaralandır
    unique_labels = sorted(train_df['label'].unique())
    label_map = {old: new for new, old in enumerate(unique_labels)}
    inverse_label_map = {v: k for k, v in label_map.items()}  # Tahmin için gerekli

    train_df['label'] = train_df['label'].map(label_map)
    test_df['label'] = test_df['label'].map(label_map)

    # Girişleri ve etiketleri ayır
    X_train = train_df.drop('label', axis=1).values.reshape(-1, *IMAGE_SIZE, 1) / 255.0
    y_train = to_categorical(train_df['label'], num_classes=CLASSES)

    X_test = test_df.drop('label', axis=1).values.reshape(-1, *IMAGE_SIZE, 1) / 255.0
    y_test = to_categorical(test_df['label'], num_classes=CLASSES)

    return X_train, y_train, X_test, y_test, label_map, inverse_label_map

def build_model():
    """Geliştirilmiş CNN modelini oluşturur."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5), # Overfittingi onlemeye yardimci olur
        Dense(256, activation='relu'),
        Dense(CLASSES, activation='softmax') # Olasılık dağılımı
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    datagen = ImageDataGenerator( # Görüntü arttırma (döndürme, kaydırma, yakınlaştırma.)
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(X_train)

    callbacks = [ # En iyi doğrulukta modeli kaydet
        ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.keras'), 
                        save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=3, restore_best_weights=True) # Gelişme durursa durdur
    ]

    history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        epochs=EPOCHS,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks)
    
    model.save(os.path.join(MODEL_DIR, 'final_model.keras'))
    return history

def evaluate_model(model, X_test, y_test, label_map):
    """Modelin başarımını ölçer."""
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    class_names = [chr(label + 65) for label in sorted(label_map.keys()) if label not in [9, 25]]

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix(y_true, y_pred), 
                annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.show()

def plot_history(history):
    """Eğitim geçmişini çizer."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

def main():
    X_train, y_train, X_test, y_test, label_map, inverse_label_map = load_and_preprocess_data()

    model = build_model()
    model.summary()

    history = train_model(model, X_train, y_train, X_test, y_test)

    plot_history(history)
    evaluate_model(model, X_test, y_test, label_map)

if __name__ == "__main__":
    main()
