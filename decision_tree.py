import numpy as np
import time

from classificator import Classificator
from decision_tree_node import Node


class DecisionTree(Classificator):
    def __init__(self, depth: int):
        self.root = Node(depth)
        self.depth = depth
        self.train_time = 0

    def train(self, x_train: np.ndarray, y_train: np.ndarray, selected_features: np.ndarray = None):
        start = time.time()
        ##########
        self.root.depth = self.depth
        self.root.train(x_train, y_train, selected_features)
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
