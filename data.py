import numpy as np
import pandas as pd


def get_wine():
    # https://archive.ics.uci.edu/dataset/186/wine+quality
    data = pd.read_csv(filepath_or_buffer='wine+quality\winequality-white.csv', delimiter=';')

    # mapping wine quality to binary values, 1 - 67%, 0 - 33% 
    data["quality"] = [1 if quality > 5 else 0 for quality in data["quality"]]

    # train data - 80%
    test_idx = np.random.choice(range(data.shape[0]), round(0.2 * data.shape[0]), replace=False)

    data_test = data.iloc[test_idx, :]
    data_train = data.drop(test_idx, axis=0)

    x_train = data_train.drop("quality", axis=1).to_numpy()
    y_train = data_train["quality"].to_numpy()
    x_test = data_test.drop("quality", axis=1).to_numpy()
    y_test = data_test["quality"].to_numpy()
    return (x_train, y_train), (x_test, y_test)


def wine_feature_names():
    return ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates",
            "alcohol"]


def get_mushrooms():
    # https://archive.ics.uci.edu/dataset/73/mushroom
    data = pd.read_csv(filepath_or_buffer='mushroom\\agaricus-lepiota.data', header=None)

    # mapping edibility to binary values, 1 - 52%, 0 - 48% 
    data.iloc[:, 0] = [1 if edible == 'e' else 0 for edible in data.iloc[:, 0]]

    # column 11 has a lot of missing values
    data = data.drop(11, axis=1)

    # train data - 80%
    test_idx = np.random.choice(range(data.shape[0]), round(0.2 * data.shape[0]), replace=False)

    data_test = data.iloc[test_idx, :]
    data_train = data.drop(test_idx, axis=0)

    x_train = data_train.drop(0, axis=1).to_numpy()
    y_train = data_train.iloc[:, 0].to_numpy()
    x_test = data_test.drop(0, axis=1).to_numpy()
    y_test = data_test.iloc[:, 0].to_numpy()
    return (x_train, y_train), (x_test, y_test)


def mushroom_feature_names():
    return ["cap-shape", "cap-surface", "cap-color", "bruises?", "odor", "gill-attachment",
            "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-surface-above-ring",
            "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring",
            "veil-type", "veil-color", "ring-number", "ring-type",
            "spore-print-color", "population", "habitat"]

def split_arrays(x: np.ndarray, y: np.ndarray, ratio: float):
    idx_first = np.random.choice(range(x.shape[0]), round(ratio * x.shape[0]), replace=False)

    mask_second = np.ones(x.shape[0], dtype=bool)
    mask_second[idx_first] = False
    idx_second = np.arange(x.shape[0])[mask_second]

    x_first, y_first = x[idx_first], y[idx_first]
    x_second, y_second = x[idx_second], y[idx_second]
    return (x_first, y_first), (x_second, y_second)
