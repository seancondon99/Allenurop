import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sklearn.metrics

oneTrackDir = '/Users/seancondon/Desktop/LHC_urop/catBoostClassifiers/oneTrack/allVariation'
twoTrackDir = '/Users/seancondon/Desktop/LHC_urop/catBoostClassifiers/twoTrack/allVariation'

def optimalTree():
    optimalAUC = 0
    optimalModel = None
    optEff = None
    optFpr = None

    TrackDir = twoTrackDir

    for modDir in os.listdir(TrackDir):
        if modDir != '.DS_Store':
            effDir = TrackDir + '/' + modDir + '/' + 'eff.txt'
            fprDir = TrackDir + '/' + modDir + '/' + 'fpr.txt'
            cutoffsDir = TrackDir + '/' + modDir + '/' + 'cutoffs.txt'
            with open(effDir) as f:
                content = f.readlines()
                eff = content[0]
            eff = eff[1:-1]
            eff = eff.split(',')
            effp = []
            for i in eff:
                effp.append(float(i))
            eff = effp

            with open(fprDir) as f:
                content = f.readlines()
                fpr = content[0]
            fpr = fpr[1:-1]
            fpr = fpr.split(',')
            fprp = []
            for i in fpr:
                fprp.append(float(i))
            fpr = fprp

            with open(cutoffsDir) as f:
                content = f.readlines()
                cutoffs = content[0]
            cutoffs = cutoffs[1:-1]
            cutoffs= cutoffs.split(',')
            cutoffsp = []
            for i in cutoffs:
                cutoffsp.append(float(i))
            cutoffs = cutoffsp

            rocAUC = sklearn.metrics.auc(fpr,eff)
            print(rocAUC)
            print(eff[50])
            print(fpr[50])
            print('\n')
            if rocAUC > optimalAUC:
                optimalAUC = rocAUC
                optimalModel = modDir
                optEff = eff
                optFpr = fpr

    print('\n')
    print(optimalModel)
    print(optimalAUC)
    print('\n')

    print(optEff[50])
    print(optFpr[50])
    print(cutoffs[50])


def optimalPreselect():
    twoTrackDir = '/Users/seancondon/Desktop/LHC_urop/catBoostClassifiers/oneTrack/preselections'

    for dir in os.listdir(twoTrackDir):
        if dir !='.DS_Store':
            preDir = twoTrackDir+'/'+dir
            cutoffDir = preDir+'/cutoffs.txt'
            triggerDir=preDir+'/trigger.txt'
            effDir = preDir+'/eff.txt'

            with open(effDir) as f:
                content = f.readlines()
                eff = content[0]
            eff = eff[1:-1]
            eff = eff.split(',')
            effp = []
            for i in eff:
                effp.append(float(i))
            eff = effp

            with open(triggerDir) as f:
                content = f.readlines()
                fpr = content[0]
            fpr = fpr[1:-1]
            fpr = fpr.split(',')
            fprp = []
            for i in fpr:
                fprp.append(float(i))
            trigger = fprp

            with open(cutoffDir) as f:
                content = f.readlines()
                cutoffs = content[0]
            cutoffs = cutoffs[1:-1]
            cutoffs= cutoffs.split(',')
            cutoffsp = []
            for i in cutoffs:
                cutoffsp.append(float(i))
            cutoffs = cutoffsp

            #we will find the sigdet closest to trigger = 1MHz
            for i in range(len(trigger)):
                if trigger[i]<3:
                    break
            print(dir)
            print(trigger[i])
            print(eff[i])

optimalPreselect()




