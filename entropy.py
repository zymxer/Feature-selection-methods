import pandas as pd
import numpy as np
import time

# import matplotlib.pyplot as plt


feature_names = ["poisonous", "cap-shape", "cap-surface", "cap-color", "bruises?", "odor", "gill-attachment",
                 "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                 "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                 "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
X = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", header=0,
                names=feature_names)
y = X["poisonous"]  # Select target label
X.drop(['poisonous'], axis=1, inplace=True)  # Remove target label from dataset
# display(X.head())  # Show some data

y = y.map({"e": 0, "p": 1})  # Mapping the classes to zeros and ones, not strictly necessary.


# display(y.head())


def entropy(y):
    probs = []  # Probabilities of each class label
    for c in set(y):  # Set gets a unique set of values. We're iterating over each value
        num_same_class = sum(y == c)  # Remember that true == 1, so we can sum.
        p = num_same_class / len(y)  # Probability of this class label
        probs.append(p)
    return np.sum([-p * np.log2(p) for p in probs if p > 0])


def class_probability(feature, y):
    """Calculates the proportional length of each value in the set of instances"""
    probs = []
    for value in set(feature):
        select = feature == value
        y_new = y[select]
        probs.append(float(len(y_new)) / len(feature))
    return probs


def class_entropy(feature, y):
    """Calculates the entropy for each value in the set of instances"""
    ents = []
    for value in set(feature):
        select = feature == value
        y_new = y[select]
        ents.append(entropy(y_new))
    return ents


def proportionate_class_entropy(feature, y):
    """Calculates the weighted proportional entropy for a feature when splitting on all values"""
    probs = class_probability(feature, y)
    ents = class_entropy(feature, y)
    return np.sum(np.multiply(probs, ents))


def entropy_filter(X: np.ndarray, y: np.ndarray):
    start = time.time()
    ents = []
    indices = []
    threshold = 0.1
    counter = 0
    for i in range(X.shape[1]):
        c = X[:, i]
        new_entr = proportionate_class_entropy(c, y)
        ents.append((c, entropy(y) - new_entr))
    for i in range(len(ents)):
        if ents[i][1] > threshold:
            indices.append(counter)
        counter+=1
    end = time.time()
    filter_time = round(end - start, 2)
    return indices, filter_time
    # filtered_columns = [feature for feature, gain in ents if gain >= threshold]
    # #return X[:filtered_columns], filter_time
    # matrix = np.column_stack(filtered_columns)
    # return matrix, filter_time



def display_entropy(X,y):
    ents = []
    for c in X.columns:
        new_entropy = proportionate_class_entropy(X[c], y)
        # print("%s %.4f" % (c, entropy(y) - new_entropy))
        ents.append((c, entropy(y) - new_entropy))

    sorted_info_gains = sorted(ents, key=lambda x: x[1], reverse=True)
    for feature, gain in sorted_info_gains:
        print("%s: %.4f" % (feature, gain))

#display_entropy(X, y)