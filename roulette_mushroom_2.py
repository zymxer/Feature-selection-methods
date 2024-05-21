import pandas as pd
import numpy as np
from mushroom_explanation import explain_result

# Wczytanie danych
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
column_names = ["label", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
data = pd.read_csv(url, header=None, names=column_names)

def calculate_fitness(row):
    return 1 if row == 'e' else 0  # jadalny - 1, trujący - 0

# Dodanie kolumny fitness do danych
data['fitness'] = data['label'].apply(calculate_fitness)

def roulette_selection(data):
    total_fitness = data['fitness'].sum()
    probabilities = data['fitness'] / total_fitness
    selected_index = np.random.choice(data.index, p=probabilities)
    return data.loc[selected_index]


selected_individual = roulette_selection(data)
print("Selected individual:")
print(selected_individual)
# Wyświetlenie wyniku wraz z objaśnieniem
# Ładuje mega długo, nie wiem skąd ten problem
#print("Selected individual with explanation:")
#print(explain_result(selected_individual))