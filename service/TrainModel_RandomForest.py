from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold

def testTrain(X = [], y = [], fold = 10, trees = 100):
    clf = RandomForestClassifier(n_estimators=trees)
    # clf = BaggingClassifier(n_estimators=100)
    score = cross_val_score(clf, X, y, cv=fold)
    print(score)
    meanAcc = 0
    for i in score:
        meanAcc = meanAcc + i
    meanAcc = (meanAcc / fold) * 100
    print(meanAcc)

def stratifiedKFold(fold=10, X=[], Y=[]):
    rf = RandomForestClassifier(n_estimators=100)
    cv = StratifiedKFold(fold)
    score, permutation_scores, pvalue = permutation_test_score(rf, X, Y, scoring="accuracy", cv=cv)
    print("Classification score %s (pvalue : %s) %s" % (score * 100, pvalue, permutation_scores))

def kFold(fold=10, X=[], Y=[]):
    rf = RandomForestClassifier(n_estimators=100)
    cv = KFold(fold)
    score, permutation_scores, pvalue = permutation_test_score(rf, X, Y, scoring="accuracy", cv=cv)
    print("Classification score %s (pvalue : %s) %s" % (score * 100, pvalue, permutation_scores))

def stratifiedShuffleSplit(fold=10, X=[], Y=[]):
    rf = RandomForestClassifier(n_estimators=100)
    cv = StratifiedShuffleSplit(fold)
    score, permutation_scores, pvalue = permutation_test_score(rf, X, Y, scoring="accuracy", cv=cv, n_permutations=10)
    print("Classification score %s (pvalue : %s) %s" % (score * 100, pvalue, permutation_scores))

def repeatedStratifiedKFold(fold=10, X=[], Y=[]):
    rf = RandomForestClassifier(n_estimators=100)
    cv = RepeatedStratifiedKFold(fold)
    score, permutation_scores, pvalue = permutation_test_score(rf, X, Y, scoring="accuracy", cv=cv, n_permutations=10)
    print("Classification score %s (pvalue : %s) %s" % (score * 100, pvalue, permutation_scores))