import numpy as np
import pandas as pd
import random
from data import get_mushrooms
def fitness(row):
    return row['label']  # Kolumna 'label' to kolumna "edibility", 1 - jadalny, 0 - trujący

def add_fitness_column(data):
    data['fitness'] = data.apply(fitness, axis=1)
    return data

def roulette_selection(data):
    total_fitness = sum(data['fitness'])
    probabilities = data['fitness'] / total_fitness
    selected_index = random.choices(data.index, weights=probabilities, k=1)[0]
    return data.loc[selected_index]


# Funkcja łączenia danych i etykiet w jeden DataFrame
def combine_data_and_labels(x, y):
    data = pd.DataFrame(x)
    data['label'] = y
    return data

if __name__ == "__main__":
    # Pobranie danych o grzybach z oryginalnego skryptu
    (x_train, y_train), (x_test, y_test) = get_mushrooms()

    # Połączenie danych wejściowych i etykiet w jeden DataFrame
    data_train = combine_data_and_labels(x_train, y_train)

    # Dodanie kolumny fitness do danych treningowych
    data_train = add_fitness_column(data_train)

    # Wybór jednego osobnika za pomocą selekcji ruletkowej
    selected_individual = roulette_selection(data_train)
    print("Selected individual:")
    print(selected_individual)
