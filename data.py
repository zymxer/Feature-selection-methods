import numpy as np
import pandas as pd

def get_wine():
    data = pd.read_csv(filepath_or_buffer='wine+quality\winequality-white.csv', delimiter=';')

    # mapping wine quality to binary values, 1 - 67%, 0 - 33% 
    data["quality"] = [1 if quality > 5 else 0 for quality in data["quality"]]

    # train data - 80%
    test_idx = np.random.choice(range(data.shape[0]), round(0.2*data.shape[0]), replace=False)

    data_test = data.iloc[test_idx, :]
    data_train = data.drop(test_idx, axis=0)

    x_train = data_train.drop("quality", axis=1).to_numpy()
    y_train = data_train["quality"].to_numpy()
    x_test = data_test.drop("quality", axis=1).to_numpy()
    y_test = data_test["quality"].to_numpy()
    return (x_train, y_train), (x_test, y_test)

def get_mushrooms():
    data = pd.read_csv(filepath_or_buffer='mushroom\\agaricus-lepiota.data', header=None)
    
    # mapping edibility to binary values, 1 - 52%, 0 - 48% 
    data.iloc[:, 0] = [1 if edible == 'e' else 0 for edible in data.iloc[:, 0]]
    
    # column 11 has a lot of missing values
    data = data.drop(11, axis=1)

    # train data - 80%
    test_idx = np.random.choice(range(data.shape[0]), round(0.2*data.shape[0]), replace=False)

    data_test = data.iloc[test_idx, :]
    data_train = data.drop(test_idx, axis=0)

    x_train = data_train.drop(0, axis=1).to_numpy()
    y_train = data_train.iloc[:, 0].to_numpy()
    x_test = data_test.drop(0, axis=1).to_numpy()
    y_test = data_test.iloc[:, 0].to_numpy()
    return (x_train, y_train), (x_test, y_test)
