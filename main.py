from decision_tree import DecisionTree
from permutation_importance import *
from data import *

# Permutation importance test
train, test = get_wine()

dec_tree = DecisionTree(6)
dec_tree.train(train[0], train[1])
print("Test result for all features: ", dec_tree.evaluate(test[0], test[1]), ". Time: ", dec_tree.train_time)

dec_tree2 = DecisionTree(6)
importance, imp_time = get_importance(dec_tree2, train[0], train[1], 20)
print("Czas wyznaczenia wpływu cech: ", imp_time)
selected_features = select_features(importance, 0.2)
dec_tree2.train(train[0], train[1], selected_features)
print("Test result for selected features: ", dec_tree2.evaluate(test[0], test[1]), ". Time: ", dec_tree2.train_time)
