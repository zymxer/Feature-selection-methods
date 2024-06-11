import numpy as np
import time
from decision_tree import DecisionTree
from classificator import Classificator

class RandomForest(Classificator):
    def __init__(self, depth: int, trees_amount: int):
        self.depth = depth
        self.trees_amount = trees_amount
        self.train_time = 0
        self.forest = []
        


    def train(self, x_train: np.ndarray, y_train: np.ndarray, selected_features: np.ndarray = None):
        start = time.time()
        for _ in range(self.trees_amount):
            x_bagging, y_bagging = self.bagging(x_train, y_train)
            tree = DecisionTree(self.depth)
            tree.train(x_bagging, y_bagging, selected_features)
            self.forest.append(tree)
        end = time.time()
        self.train_time = round(end - start, 2)

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        """ Returns model accuracy in % """
        prediction = self.predict(x)
        return round(np.mean(prediction == y) * 100, 2)

    def predict(self, x: np.ndarray) -> list:
        tree_predictions = []
        for tree in self.forest:
            tree_predictions.append(tree.predict(x))

        array = np.array(tree_predictions)
        predicted = np.mean(array, axis=0)
        predicted = array.tolist()
        return predicted

    def bagging(self, x_train: np.ndarray, y_train: np.ndarray):
        selected_indices = np.random.choice(x_train.shape[0], size=int(x_train.shape[0]), replace=True)
        x_selected = x_train[selected_indices, :]
        y_selected = y_train[selected_indices]
        return x_selected, y_selected
