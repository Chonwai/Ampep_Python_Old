import os
from service import GetFeature as GetFeature

def main():
    GetFeature.getFeature('./data/trian_po_set3298_for_ampep_sever.fasta', './data/trian_po_set3298_for_ampep_sever.tsv', 'AAC')

main()