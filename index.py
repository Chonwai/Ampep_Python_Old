import os
import numpy as np
import sys
import timeit
from service import GetFeature as GetFeature
from service import TrainingMethod as RFTrainer
from service import Utils as Utils

feature = sys.argv[1]
ml = sys.argv[2]
method = sys.argv[3]
fold = int(sys.argv[4])
trees = int(sys.argv[5])

def main():
    # GetFeature.getFeature('./data/trian_po_set3298_for_ampep_sever.fasta',
    #                       './data/trian_po_set3298_for_ampep_sever.tsv', feature)
    # GetFeature.getFeature('./data/trian_ne_set9894_for_ampep_sever.fasta',
    #                       './data/trian_ne_set9894_for_ampep_sever.tsv', feature)
    utils = Utils.Utils('Train')
    posArray, posY = utils.readFeature(
        "data/trian_po_set3298_for_ampep_sever.tsv", 1)
    negArray, negY = utils.readFeature(
        "data/trian_ne_set9894_for_ampep_sever.tsv", 0)
    X = np.concatenate((posArray, negArray))
    y = np.concatenate((posY, negY))
    print(len(X), len(y))

    trainer = RFTrainer.Trainer(X, y)
    trainer.training(fold, trees, method)

start = timeit.default_timer()
main()
stop = timeit.default_timer()
print('Time: ', stop - start)