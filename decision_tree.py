import numpy as np
import time

from decision_tree_node import Node

class DecisionTree:
    def __init__(self, depth: int):
        self.root = Node(depth)
        self.depth = depth
        self.train_time = 0

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        start = time.time()
        self.root.train(x_train, y_train)
        end = time.time()
        self.train_time = round(end - start, 2)

    def predict(self, x: np.ndarray) -> list:
        prediction = []
        for observation in x:
            prediction.append(0 if self.root.predict(observation) <= 0.5 else 1)
        return prediction

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """ Returns model accuracy in % """
        prediction = self.predict(x)
        return round(np.mean(prediction == y) * 100, 2)
