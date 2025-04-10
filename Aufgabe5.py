import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import cv2
import os

# Buchstaben-Map
buchstaben = [chr(i) for i in range(ord('A'), ord('Z')+1)]

# Funktion zum Laden der Daten
def load_data(input_folder, size=(28, 28)):
    images = []
    labels = []
    for label, letter in enumerate(sorted(os.listdir(input_folder))):
        letter_path = os.path.join(input_folder, letter)
        if os.path.isdir(letter_path):
            for filename in os.listdir(letter_path):
                if filename.lower().endswith(".png"):
                    img_path = os.path.join(letter_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                    img_array = img_resized / 255.0
                    images.append(img_array)
                    labels.append(label)

    x_data = np.array(images).reshape(-1, 28, 28, 1).astype('float32')
    y_data = np.array(labels).astype('int')
    return x_data, y_data

# Daten laden und aufteilen
x, y = load_data("C:\\Users\\LENOVO\\OneDrive - HTL Anichstrasse\\Dokumente\\4cHel\\KI\\VEN\\BigDataSet")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Modell laden
model = load_model("train_model.keras")

# Funktion zur Vorhersageanzeige
def predict(model, x_test, y_test, anzahl=50):
    predictions = model.predict(x_test[:anzahl])
    for i in range(anzahl):
        wahrscheinlichkeit = predictions[i] # Vorhersage-Wahrscheinlichkeiten
        vorhergesagt_index = np.argmax(wahrscheinlichkeit) # Vorhergesagter Index
        wahr_index = y_test[i] # Wahrer Index

        print(f"Bild {i+1}:")
        print(f" → Wahr: {buchstaben[wahr_index]}")
        print(f" → Vorhergesagt: {buchstaben[vorhergesagt_index]}")
        print(f" → Wahrscheinlichkeit: {wahrscheinlichkeit[vorhergesagt_index]:.4f}")
        print("")

# Funktion aufrufen
predict(model, x_test, y_test, anzahl=50)
model.save("verbessertes_cnn_model.keras")
print("Modell gespeichert als verbessertes_cnn_model.keras") 
