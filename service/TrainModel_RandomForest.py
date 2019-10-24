from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

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
