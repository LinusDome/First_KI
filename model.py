import random

class SimpleNeuralNetwork:
    def __init__(self):
        self.weight = random.uniform(-1, 1)  # Zufälliges Startgewicht

    def forward(self, x):
        return x * self.weight  # Einfaches lineares Modell

    def train(self, data, labels, lr=0.01, epochs=100):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(data)):
                x, y_true = data[i], labels[i]
                y_pred = self.forward(x)
                loss = (y_pred - y_true) ** 2  # Fehler berechnen
                total_loss += loss
                grad = 2 * (y_pred - y_true) * x  # Ableitung des Fehlers
                self.weight -= lr * grad  # Gewichtsupdate
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    def predict(self, x):
        return self.forward(x)