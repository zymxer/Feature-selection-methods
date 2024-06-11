import time
from classificator import Classificator
from data import *


def get_importance(cf: Classificator, x, y, shuffles: int) -> tuple:
    start = time.time()
    # splits arrays to use test part to measure importance
    train, test = split_arrays(x, y, 0.6)

    x_train, y_train = train[0], train[1]
    x_test, y_test = test[0], test[1]

    cf.train(x_train, y_train)
    ref_score = cf.evaluate(x_test, y_test)

    importance = []

    for i in range(x_train.shape[1]):
        scores = []
        for j in range(shuffles):
            shuffled_x = permute_feature(x_test, i)
            scores.append(cf.evaluate(shuffled_x, y_test))
        score = np.mean(scores)
        importance.append(ref_score - score)
    end = time.time()
    return importance, round(end - start, 2)


def permute_feature(x_test: np.ndarray, feature: int) -> np.ndarray:
    shuffled = np.copy(x_test)
    np.random.shuffle(shuffled[:, feature])
    return shuffled


def present_importance(importance: list, feature_names):
    for i in range(len(importance)):
        print("Importance for feature ", feature_names[i], ":\t", round(importance[i], 2))


def select_features(importance: list, fraction: float) -> np.ndarray:
    importance_array = np.array(importance)
    sorted_indices = np.argsort(importance_array)[::-1]
    features_to_keep = max(int(len(importance) * fraction), 1)
    return sorted_indices[:features_to_keep]
