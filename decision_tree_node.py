import numpy as np

class Node:
    def __init__(self, depth: int):
        self.left = None
        self.right = None
        self.depth = depth
        self.feature_idx = None
        self.feature_value = None
        self.prediction = None

    def gini_best_score(self, y_train: np.ndarray, splits: list):
        """ Returns index and score of best split for given SORTED label """

        best_gain = -np.inf
        best_idx = 0

        # calculates score for each possible split
        for split in splits:
            left_pos, left_neg = 0, 0
            right_pos, right_neg = 0, 0
            for i in range(y_train.size):
                if i < split + 1:
                    if y_train[i] == 1:
                        left_pos += 1
                    else:
                        left_neg += 1
                else:
                    if y_train[i] == 1:
                        right_pos += 1
                    else:
                        right_neg += 1

            gini_left = 1 - (left_pos / (left_pos + left_neg)) ** 2 - (left_neg / (left_pos + left_neg)) ** 2
            gini_right = 1 - (right_pos / (right_pos + right_neg)) ** 2 - (right_neg / (right_pos + right_neg)) ** 2

            left = left_pos + left_neg
            right = right_pos + right_neg

            gini_gain = 1 - (left / (left + right)) * gini_left - (right / (left + right)) * gini_right

            if gini_gain > best_gain:
                best_gain = gini_gain
                best_idx = split

        return best_idx, best_gain

    def possible_splits(self, feature: np.ndarray) -> list:
        """ Returns possible splits for given SORTED feature """

        splits = []
        for i in range(feature.size - 1):
            if feature[i] != feature[i + 1]:
                splits.append(i)
        return splits

    def best_split(self, x_train: np.ndarray, y_train: np.ndarray, selected_features: np.ndarray = None):
        """ Returns best split for given dataset """

        best_gain = -np.inf
        best_split = None

        features = range(x_train.shape[1]) if selected_features is None else selected_features

        for i in features:
            feature = x_train[:, i]

            order = np.argsort(feature)

            possible_splits = self.possible_splits(feature[order])

            feature_value_idx, feature_gain = self.gini_best_score(y_train[order], possible_splits)
            if feature_gain > best_gain:
                best_gain = feature_gain
                # best_split = (feature_index, feature_value)
                best_split = (i, feature[order[feature_value_idx]])

        # no mean value for split, splitting categorical values as <= and > parts
        if best_split is None:
            return None, None
        return best_split[0], best_split[1]

    def split_dataset(self, x_train: np.ndarray, y_train: np.ndarray, feature_idx: int, value):
        """ Splits dataset into left and right parts by given feature and value to split by """

        left_mask = x_train[:, feature_idx] <= value
        return (x_train[left_mask], y_train[left_mask]), (x_train[~left_mask], y_train[~left_mask])

    def train(self, x_train: np.ndarray, y_train: np.ndarray, selected_features: np.ndarray = None):

        self.prediction = np.mean(y_train)
        # if 1 observation is left or prediction is exact
        if x_train.shape[0] == 1 or self.prediction == 0 or self.prediction == 1:
            return

        self.feature_idx, self.feature_value = self.best_split(x_train, y_train, selected_features)
        if self.feature_idx is None:
            return

        (x_left, y_left), (x_right, y_right) = self.split_dataset(x_train, y_train, self.feature_idx, self.feature_value)

        if self.depth == 0:
            return

        self.left, self.right = Node(self.depth - 1), Node(self.depth - 1)
        self.left.train(x_left, y_left, selected_features)
        self.right.train(x_right, y_right, selected_features)

    def predict(self, observation: np.ndarray):
        """ Predicts the class of given observation """
        # leaf case
        if self.left is None and self.right is None:
            return self.prediction
        if observation[self.feature_idx] <= self.feature_value:
            return self.left.predict(observation)
        else:
            return self.right.predict(observation)