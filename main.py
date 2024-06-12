from decision_tree import DecisionTree
from random_forest import RandomForest
from permutation_importance import *
from entropy import *
from data import *
import numpy as np

# Permutation importance test
train, test = get_wine()
names = wine_feature_names()
#cl1 = RandomForest(2, 3)
cl1 = DecisionTree(5)
#cl2 = RandomForest(2, 3)
cl2 = DecisionTree(5)

cl1.train(train[0], train[1])
print("Wynik dla wszystkich cech: ", cl1.evaluate(test[0], test[1]), ". Czas treningu: ", cl1.train_time)
print("-----------------------------")

print("Permutation importance:")
importance, imp_time = get_importance(cl2, train[0], train[1], 20)
print("Czas wyznaczenia wpływu cech: ", imp_time)
selected_features = select_features(importance, 0.2)

print("\nWybrane cechy:")
for i in range(len(selected_features)):
    print(names[selected_features[i]])

cl2.train(train[0], train[1], selected_features)
print("\nWynik dla wybranych cech: ", cl2.evaluate(test[0], test[1]), ". Czas: ", cl2.train_time)
print("-----------------------------")

print("Entropy:")
selected, filter_time = entropy_filter(train[0], train[1])
print("Czas filtracji cech: ", filter_time)

print("\nWybrane cechy:")
for i in range(len(selected)):
    print(names[selected[i]])
cl2.train(train[0], train[1], selected)
print("\nWynik dla wybranych cech: ", cl2.evaluate(test[0], test[1]), ". Czas: ", cl2.train_time)
