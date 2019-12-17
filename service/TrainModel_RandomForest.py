from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef
import pickle

class Trainer():
    def __init__(self, X=[], y=[]):
        self.X = X
        self.y = y

    def training(self, fold=10, trees=100, method="KFold"):
        for i in range(1):
            clf = RandomForestClassifier(n_estimators=(trees + 100 * i))
            cv = self.router(method, fold)
            
            score = cross_validate(clf.fit(self.X, self.y), self.X, self.y, cv=cv, n_jobs=-1, scoring=('accuracy', 'average_precision'))
            # score = cross_val_predict(clf, self.X, self.y, cv=cv, n_jobs=-1, scoring=('accuracy', 'average_precision'))
            print("Finished Training Model " + str(i) + " Times with " + str(fold) + " Fold and " + str(trees + 100 * i) + " Trees!")

            model = './model/' + method + '/RF_' + method + '_' + str(fold) + '_' + str(trees + 100 * i) + '.pkl'

            with open(model, 'wb') as file:
                pickle.dump(clf, file)

            self.calculateScore(score, fold, scoring=['accuracy', 'average_precision'])

    def calculateScore(self, score, fold, scoring):
        for i in scoring:
            name = 'test_' + i
            print(name)
            meanAcc = 0
            for i in score[name]:
                meanAcc = meanAcc + i
            meanAcc = (meanAcc / fold) * 100
            print("Mean Accuracy: " + str(meanAcc))
            print("Top Accuracy: " + str(max(score[name]) * 100))
            print("Bottom Accuracy: " + str(min(score[name]) * 100))
            print('\n')

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