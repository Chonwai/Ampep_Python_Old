from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
import pickle

class Trainer():
    def __init__(self, X=[], y=[]):
        self.X = X
        self.y = y

    def training(self, fold=10, trees=100, method="KFold"):
        for i in range(5):
            clf = RandomForestClassifier(n_estimators=(trees + 100 * i))
            cv = self.router(method, fold)
            score = cross_validate(clf, self.X, self.y, cv=cv)
            print("Finished Training Model " + str(i) + " Times with " + str(trees + 100 * i) + " Trees!")
            print(score)

            model = './model/RepeatedStratifiedKFold/RF_' + method + '_0' + str(i) + '.pkl'
            with open(model, 'wb') as file:
                pickle.dump(score, file)

            meanAcc = 0
            for i in score['test_score']:
                meanAcc = meanAcc + i
            meanAcc = (meanAcc / fold) * 100
            print("Mean Accuracy: " + str(meanAcc))
            print("Top Accuracy: " + str(max(score['test_score']) * 100))
            print("Bottom Accuracy: " + str(min(score['test_score']) * 100))

    def router(self, method="KFold", fold=10):
        if (method == "KFold"):
            cv = KFold(n_splits=fold)
        elif (method == "StratifiedKFold"):
            cv = StratifiedKFold(n_splits=fold)
        elif (method == "StratifiedShuffleSplit"):
            cv = StratifiedShuffleSplit(n_splits=fold)
        elif (method == "RepeatedStratifiedKFold"):
            cv = RepeatedStratifiedKFold(n_splits=fold)
        else:
            cv = KFold(n_splits=fold)
        return cv

# def testTrain(X = [], y = [], fold = 10, trees = 100, method="KFold"):
#     clf = RandomForestClassifier(n_estimators=trees)

#     if (method == "KFold"):
#         cv = KFold(fold)
#     else:
#         cv = KFold(fold)

#     score = cross_val_score(clf, X, y, cv=cv)
#     print(score)
#     meanAcc = 0
#     for i in score:
#         meanAcc = meanAcc + i
#     meanAcc = (meanAcc / fold) * 100
#     print(meanAcc)

# def stratifiedKFold(fold=10, X=[], Y=[]):
#     rf = RandomForestClassifier(n_estimators=100)
#     cv = StratifiedKFold(fold)
#     score, permutation_scores, pvalue = permutation_test_score(rf, X, Y, scoring="accuracy", cv=cv)
#     print("Classification score %s (pvalue : %s) %s" % (score * 100, pvalue, permutation_scores))

# def kFold(fold=10, X=[], Y=[]):
#     rf = RandomForestClassifier(n_estimators=100)
#     cv = KFold(fold)
#     score, permutation_scores, pvalue = permutation_test_score(rf, X, Y, scoring="accuracy", cv=cv)
#     print("Classification score %s (pvalue : %s) %s" % (score * 100, pvalue, permutation_scores))

# def stratifiedShuffleSplit(fold=10, X=[], Y=[]):
#     rf = RandomForestClassifier(n_estimators=100)
#     cv = StratifiedShuffleSplit(fold)
#     score, permutation_scores, pvalue = permutation_test_score(rf, X, Y, scoring="accuracy", cv=cv, n_permutations=10)
#     print("Classification score %s (pvalue : %s) %s" % (score * 100, pvalue, permutation_scores))

# def repeatedStratifiedKFold(fold=10, X=[], Y=[]):
#     rf = RandomForestClassifier(n_estimators=100)
#     cv = RepeatedStratifiedKFold(fold)
#     score, permutation_scores, pvalue = permutation_test_score(rf, X, Y, scoring="accuracy", cv=cv, n_permutations=10)
#     print("Classification score %s (pvalue : %s) %s" % (score * 100, pvalue, permutation_scores))
