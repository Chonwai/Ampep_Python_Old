import os
import numpy as np
import sys
from service import GetFeature as GetFeature
from service import TrainModel_RandomForest as TestTrain
from service import Router as Router
from service import Utility as Utility

feature = sys.argv[1]
method = sys.argv[2]

def main():
    GetFeature.getFeature('./data/trian_po_set3298_for_ampep_sever.fasta',
                          './data/trian_po_set3298_for_ampep_sever.tsv', feature)
    GetFeature.getFeature('./data/trian_ne_set9894_for_ampep_sever.fasta',
                          './data/trian_ne_set9894_for_ampep_sever.tsv', feature)
    utility = Utility.Utility('Train')
    posArray, posY = utility.readFeature(
        "data/trian_po_set3298_for_ampep_sever.tsv", 1)
    negArray, negY = utility.readFeature(
        "data/trian_ne_set9894_for_ampep_sever.tsv", 0)
    X = np.concatenate((posArray, negArray))
    y = np.concatenate((posY, negY))
    print(len(X), len(y))

    router = Router.Router(method)
    router.randomForest(X, y, int(sys.argv[3]), int(sys.argv[4]))

main()
