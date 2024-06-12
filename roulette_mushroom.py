import pandas as pd
import numpy as np
import time

# Pomiar czasu wczytywania i przetwarzania danych
start_time = time.time()

# Wczytywanie zbioru danych
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
           'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
           'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
           'stalk-surface-below-ring', 'stalk-color-above-ring',
           'stalk-color-below-ring', 'veil-type', 'veil-color',
           'ring-number', 'ring-type', 'spore-print-color', 'population',
           'habitat']

mushroom_data = pd.read_csv(url, header=None, names=columns)

# Przetwarzanie atrybutów 
label_encoders = {}
for column in mushroom_data.columns:
    le = {val: idx for idx, val in enumerate(mushroom_data[column].unique())}
    mushroom_data[column] = mushroom_data[column].map(le)
    label_encoders[column] = le

data_processing_time = time.time()
print(f"Przetwarzanie danych zajęło: {data_processing_time - start_time:.4f} sekundy")

# Dane
X = mushroom_data.drop('class', axis=1).values
y = mushroom_data['class'].values

# Ręczne dzielenie na zbiór treningowy i testowy
def train_test_split_manual(X, y, test_size=0.2):
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split_manual(X, y)

data_split_time = time.time()
print(f"Podział danych na treningowe i testowe zajęło: {data_split_time - data_processing_time:.4f} sekundy")

# Implementacja prostej klasyfikacji drzewa decyzyjnego
class SimpleDecisionTree:
    def __init__(self):
        self.tree = {}

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts[np.nonzero(counts)] / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, X, y, feature):
        parent_entropy = self._entropy(y)
        values, counts = np.unique(X[:, feature], return_counts=True)
        weighted_entropy = np.sum([
            (counts[i] / len(y)) * self._entropy(y[X[:, feature] == v])
            for i, v in enumerate(values)
        ])
        return parent_entropy - weighted_entropy

    def _best_feature_to_split(self, X, y):
        features = X.shape[1]
        gains = [self._information_gain(X, y, f) for f in range(features)]
        return np.argmax(gains)

    def _build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return y[0]
        if X.shape[1] == 0:
            return np.bincount(y).argmax()

        best_feature = self._best_feature_to_split(X, y)
        tree = {best_feature: {}}
        for value in np.unique(X[:, best_feature]):
            sub_X = X[X[:, best_feature] == value]
            sub_y = y[X[:, best_feature] == value]
            subtree = self._build_tree(np.delete(sub_X, best_feature, axis=1), sub_y)
            tree[best_feature][value] = subtree

        return tree

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_one(self, tree, sample):
        if not isinstance(tree, dict):
            return tree
        feature = next(iter(tree))
        value = sample[feature]
        subtree = tree[feature].get(value, np.bincount(y_train).argmax())
        return self._predict_one(subtree, sample)

    def predict(self, X):
        return np.array([self._predict_one(self.tree, sample) for sample in X])

    def predict_proba(self, X):
        predictions = self.predict(X)
        probs = np.zeros((len(predictions), 2))
        for i, pred in enumerate(predictions):
            if pred == 0:
                probs[i] = [1, 0]
            else:
                probs[i] = [0, 1]
        return probs

model = SimpleDecisionTree()
model.fit(X_train, y_train)

training_time = time.time()
print(f"Trening modelu zajęło: {training_time - data_split_time:.4f} sekundy")

# Przewidywanie na zbiorze testowym i obliczanie dopasowania
y_pred_proba = model.predict_proba(X_test)
fitness_values = np.max(y_pred_proba, axis=1)

prediction_time = time.time()
print(f"Przewidywanie i obliczanie dopasowania zajęło: {prediction_time - training_time:.4f} sekundy")

# Normalizacja
total_fitness = np.sum(fitness_values)
probabilities = fitness_values / total_fitness

# Funkcja do selekcji ruletkowej
def roulette_wheel_selection(probabilities, num_selections):
    cumulative_prob = np.cumsum(probabilities)
    selections = []
    for _ in range(num_selections):
        r = np.random.rand()
        for i, prob in enumerate(cumulative_prob):
            if r <= prob:
                selections.append(i)
                break
    return selections

# Przeprowadzenie selekcji (wybór 10 próbek)
num_selections = 10
selected_indices = roulette_wheel_selection(probabilities, num_selections)
selected_samples = X_test[selected_indices]

selection_time = time.time()
print(f"Selekcja ruletkowa zajęła: {selection_time - prediction_time:.4f} sekundy")

print("Wybrane próbki:\n", selected_samples)
