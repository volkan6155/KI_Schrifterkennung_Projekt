import os
import numpy as np
import cv2
import matplotlib.pyplot as plt



def load_data(input_folder, size=(28, 28)):
    images = []
    labels = []

    for label, letter in enumerate(sorted(os.listdir(input_folder))): 
        letter_path = os.path.join(input_folder, letter)#Erstellt den Pfad zu den einzelnen Buchstaben
        if os.path.isdir(letter_path): 
            for filename in os.listdir(letter_path):
                if filename.lower().endswith(".png"):
                    img_path = os.path.join(letter_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                    img_array = img_resized / 255.0  # Normalize pixel values
                    images.append(img_array)
                    labels.append(label)


    return np.array(images), np.array(labels).reshape(-1, 1)

##Beispielaufruf:
X, y = load_data("C:\\Users\\LENOVO\\OneDrive - HTL Anichstrasse\\Dokumente\\4cHel\\KI\\VEN\\BigDataSet")
plt.imshow(X[15], cmap='gray')  # Verwenden von 'gray' für Graustufenbilder
plt.axis('off')  # Entfernt die Achsen
plt.show()
print(X.shape, y.shape)
print(y)