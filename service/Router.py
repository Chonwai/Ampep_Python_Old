from service import TrainModel_RandomForest as TrainRandomForest

class Router():
    def __init__(self, method = ''):
        self.method = method
    def randomForest(self, X, y, fold = 10, trees = 100):
        # TrainRandomForest.testTrain(X, y, fold, trees)
        # TrainRandomForest.stratifiedKFold(X=X, Y=y)
        # TrainRandomForest.kFold(X=X, Y=y)
        TrainRandomForest.stratifiedShuffleSplit(X=X, Y=y)
        # TrainRandomForest.repeatedStratifiedKFold(X=X, Y=y)