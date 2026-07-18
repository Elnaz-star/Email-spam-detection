import math
import random

class LogisticRegression:
    def __init__(self, learning_rate=0.05, epochs=500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def train(self, X, y):
        features = len(X[0])
        self.weights = [0.0] * features

        for _ in range(self.epochs):
            gradients = [0.0] * features
            bias_gradient = 0

            for row, label in zip(X, y):
                prediction = self.sigmoid(sum(a*b for a,b in zip(row,self.weights)) + self.bias)
                error = prediction - label

                for i in range(features):
                    gradients[i] += error * row[i]
                bias_gradient += error

            size = len(X)
            for i in range(features):
                self.weights[i] -= self.learning_rate * gradients[i] / size

            self.bias -= self.learning_rate * bias_gradient / size

    def predict_probability(self, X):
        result=[]
        for row in X:
            value=sum(a*b for a,b in zip(row,self.weights))+self.bias
            result.append(self.sigmoid(value))
        return result

    def predict(self, X):
        return [1 if p >= .5 else 0 for p in self.predict_probability(X)]
