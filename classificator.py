from abc import ABC, abstractmethod
import numpy as np


class Classificator(ABC):
    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> list:
        pass

    @abstractmethod
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        pass
