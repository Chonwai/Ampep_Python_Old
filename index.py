import os
import numpy as np
from service import GetFeature as GetFeature
from service import TrainModel_RandomForest as TestTrain
from service import Utility as Utility

def main():
    GetFeature.getFeature('./data/trian_po_set3298_for_ampep_sever.fasta', './data/trian_po_set3298_for_ampep_sever.tsv', 'CTDC')
    GetFeature.getFeature('./data/trian_ne_set9894_for_ampep_sever.fasta', './data/trian_ne_set9894_for_ampep_sever.tsv', 'CTDC')
    utility = Utility.Utility('Train')
    posArray, posY = utility.readFeature("data/trian_po_set3298_for_ampep_sever.tsv", 1)
    negArray, negY = utility.readFeature("data/trian_ne_set9894_for_ampep_sever.tsv", 0)
    X = np.concatenate((posArray, negArray))
    y = np.concatenate((posY, negY))
    print(len(X), len(y))
    TestTrain.testTrain(X, y)
    
main()