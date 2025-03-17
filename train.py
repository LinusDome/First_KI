from model import SimpleNeuralNetwork

# Beispiel-Daten (Eingabe, gewünschte Ausgabe)
data = [1, 2, 3, 4, 5]
labels = [2, 4, 6, 8, 10]  # Ein einfaches Muster: y = 2x

# KI-Modell erstellen und trainieren
nn = SimpleNeuralNetwork()
nn.train(data, labels)

# Testen
print("Vorhersage für 6:", nn.predict(6))  # Erwartet etwa 12