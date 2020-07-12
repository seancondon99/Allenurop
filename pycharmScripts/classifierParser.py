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
    twoTrackDir = '/Users/seancondon/Desktop/LHC_urop/catBoostClassifiers/custom'

    for dir in os.listdir(twoTrackDir):
        if dir !='.DS_Store' and 'specific' not in dir:
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
                #if trigger[i]<3:
                #    break
                if (10**trigger[i])/30000 < 0.0954:
                    break
            print(dir)
            triggerNOLOG = 10 ** trigger[i]
            falsepr = triggerNOLOG/30000
            print(falsepr)
            print(eff[i])

def specificRearrange():
    twoTrackDir = '/Users/seancondon/Desktop/LHC_urop/catBoostClassifiers/custom'

    for dir in os.listdir(twoTrackDir):
        if 'specific' in dir:
            cutoffs = np.linspace(0, 1, 1001)

            sigdeteff_meta = np.zeros(len(cutoffs) - 1)
            trigger_meta = np.zeros(len(cutoffs) - 1)
            fpr_meta = np.zeros(len(cutoffs) - 1)
            added=0
            plt.figure()
            handles=[]
            for f in os.listdir(twoTrackDir+'/'+dir):
                if '.root' in f:

                    cwd = twoTrackDir+'/'+dir+'/'+f+'/'
                    with open(cwd+'eff.txt') as file:
                        content = file.readlines()
                        eff = content[0]
                    eff = eff[1:-1]
                    eff = eff.split(',')
                    effp = []
                    for i in eff:
                        effp.append(float(i))
                    eff = effp

                    with open(cwd+'trigger.txt') as file:
                        content = file.readlines()
                        trig = content[0]
                    trig = trig[1:-1]
                    trig = trig.split(',')
                    trigp = []
                    for i in trig:
                        trigp.append(float(i))
                    trig = trigp

                    sigdeteff_meta=np.add(sigdeteff_meta,eff)
                    trigger_meta=np.add(trigger_meta,trig)
                    added+=1
                    hand, = plt.step(trig,eff,label=f.split('.')[0])
                    handles.append(hand)
            plt.legend(handles=handles)
            plt.xlabel('Trigger Rate (Log KHz)')
            plt.ylabel('Signal Detection Efficiency')
            plt.title('Efficiency by Decay Type')
            plt.savefig(twoTrackDir+'/'+dir+'/effByDecay.pdf')



            sigdeteff_meta=np.divide(sigdeteff_meta,added)
            trigger_meta=np.divide(trigger_meta,added)
            plt.figure()
            plt.step(trigger_meta,sigdeteff_meta)
            plt.xlabel('Trigger Rate (Log KHz)')
            plt.ylabel('Signal Detection Efficiency')
            plt.title('Average Efficiency for '+ dir)
            plt.savefig(twoTrackDir + '/' + dir + '/avgEff.pdf')

            modeldir = twoTrackDir + '/' + dir
            with open(modeldir + '/eff.txt', 'w') as f:
                json.dump(list(sigdeteff_meta), f)
            with open(modeldir + '/trigger.txt', 'w') as f:
                json.dump(list(trigger_meta), f)



#specificRearrange()




