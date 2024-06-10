from decision_tree import DecisionTree
from data import *

dec_tree = DecisionTree(7)
train, test = get_wine()
dec_tree.train(train[0], train[1])
print(dec_tree.train_time)
print("Train result: ", dec_tree.evaluate(train[0], train[1]))
print("Test result: ", dec_tree.evaluate(test[0], test[1]))
