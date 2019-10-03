import os

def getFeature(input, output, type):
    feature = os.system('python ./iFeature/iFeature.py --file ' + input + ' --type ' + type + ' --out ' + output)
    return feature