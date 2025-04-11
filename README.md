# KI_Schrifterkennung_Projekt
Projekt: Schrifterkennungs-KI
Schritte:
1. Datensatzvorbereitung
Daten laden: Bilder der Blockbuchstaben werden aus einem Ordner geladen, skaliert und normalisiert.

Code: load_data lädt die Bilder, wandelt sie in Graustufen um und skaliert sie auf 28x28 Pixel.

2. Datenaufteilung und Augmentierung
Aufteilung: Die Daten werden in Trainings- und Testsets aufgeteilt (90% Training, 10% Test).

Datenaugmentation: Mit ImageDataGenerator werden Trainingsbilder zufällig rotiert, verschoben und skaliert.

3. Modelltraining
CNN-Modell: Ein Convolutional Neural Network (CNN) mit 3 Conv2D-Schichten und MaxPooling wird erstellt.

Regularisierung: L2-Regularisierung und Dropout werden genutzt, um Overfitting zu vermeiden.

Kompilierung: Das Modell wird mit Adam und sparse_categorical_crossentropy als Verlustfunktion kompiliert.

4. Modell evaluieren und speichern
Testen: Nach dem Training wird das Modell mit den Testdaten evaluiert.

Speichern: Das trainierte Modell wird mit model.save("train_model.keras") gespeichert.

5. Vorhersage auf Testdaten
Vorhersage: Eine Funktion berechnet die Vorhersage für jedes Testbild und vergleicht die wahre mit der vorhergesagten Klasse.

Ausgabe: Es wird der vorhergesagte Buchstabe sowie die Wahrscheinlichkeit angezeigt.

6. Benutzeroberfläche (GUI)
Zeichnen: Mit Tkinter kann der Benutzer Buchstaben zeichnen, die dann vom Modell erkannt werden.

Erkennung: Das gezeichnete Bild wird skaliert, binarisiert und dem Modell zur Vorhersage übergeben.
