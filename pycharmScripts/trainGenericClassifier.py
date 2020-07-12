import sys, os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import uproot
import math
from catboost import CatBoostClassifier, Pool
import json
import random

tree_dict = {
    'N1Trk' : 'DecayTreeTuple/N1Trk',
    'N2Trk' : 'DecayTreeTuple#1/N2Trk',
    'N3Trk' : 'DecayTreeTuple#2/N3Trk',
    'N4Trk' : 'DecayTreeTuple#3/N4Trk'
}

#custom exceptions for debugging
class customException(Exception):
    pass

try:
    import uproot_custom
except:
    raise customException('Make sure trainGenericClassifier.py is in the same directory as uproot_custom.py')

def parseArgs():
    '''
    Parses command line arguments needed for training the BDT model.
    :return: parser arguments object
    '''
    '[modelDir, modelType,learningRate, iterations, depth, dataDir]'

    parser = argparse.ArgumentParser(description='Script for creating a one-track or two-track trigger with a Boosted Decision Tree')

    parser.add_argument('--modelType',type=str, default='oneTrack',required=False,
                        help='The type of model, possible values are [oneTrack,twoTrack]')
    parser.add_argument('--modelDir',type=str, default='./catBoostClassifiers/',required=False,
                        help='Directory to saved trained model into')
    parser.add_argument('--learningRate',type=str, default='0.01',required=False,
                        help='Learning rate for BDT model, default is 0.01')
    parser.add_argument('--iterations',type=str, default='800', required=False,
                        help='Iterations for BDT model, default is 800')
    parser.add_argument('--depth', type=str,default='8', required=False,
                        help='Depth of decision trees in model, default is 8')
    parser.add_argument('--dataDir',type=str, default='AllenMVA/data', required=False,
                        help='Directory containing .root files of training data, default is ../AllenMVA/data')
    parser.add_argument('--weightingType',type=str,default='example',required=False,
                        help='Method by which training examples are weighted, possible values are [example, event]')
    parser.add_argument('--randomShuffle',type=str, default='False',required=False,
                        help='Randomly shuffle training data to avoid overtraining on certain decay types. Possible values are [True, False]')

    return parser.parse_args()

def createData(oneTrack,dataDir):
    '''
    Loads data from .root files specified by --dataDir and puts them in format for training / testing.
    75% of data is used for training, 25% for testing (can be changed).
    :param oneTrack: Boolean (True for one-track model, False for two-track model)
    :return: training and testing data and labels as lists
    '''
    #Make sure the data directory and oneTrack are valid
    if not isinstance(oneTrack,bool):
        raise customException('Please input either oneTrack or twoTrack for the --modelType argument.')
    try:
        dataFiles = os.listdir(dataDir)
    except:
        raise customException('Please use a valid directory for --dataDir argument')

    #load appropriate data for either a oneTrack or twoTrack model
    allenMVAdir = dataDir
    trunks = ['N1Trk', 'N2Trk', 'N3Trk', 'N4Trk']
    if oneTrack:
        tossed=0
        total=0
        tdataMeta = {}
        tlabelMeta = {}
        dataMeta = {}
        labelMeta = {}

        for f in dataFiles:
            if f != '2018MinBias_MVATuple.root' and f != '.DS_Store':
                eventDict = {}
                eventLabels = {}
                teventDict = {}
                teventLabels = {}

                ptvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_PT')
                ipchi2vec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_IPCHI2_OWNPV')
                sigtypevec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_signal_type')
                evinseqvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='EventInSequence')
                cutoff_index = math.floor(len(ptvec) * 0.75) #75% data used for training, this can be changed

                #VET THE PARENT PARTICLES SO WE KNOW THEY COULD BE PROPERLY RECONSTRUCTED
                #vetting code...
                truePTvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_signal_TRUEPT')
                trueTauvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_signal_TRUETAU')
                momx = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_signal_TRUEP_X')
                momy = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_signal_TRUEP_Y')
                momz = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_signal_TRUEP_Z')
                mome = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_signal_TRUEP_E')

                magp = []
                for i in range(len(momx)):
                    val = ((momx[i]) ** 2 + (momy[i]) ** 2 + (momz[i]) ** 2) ** 0.5
                    magp.append(val)
                eta = []
                for i in range(len(magp)):
                    val = math.atanh(momz[i] / magp[i])
                    eta.append(val)
                etaPreselect = []
                for i in range(len(eta)):
                    if eta[i] < 2 or eta[i] > 5:
                        etaPreselect.append(False)
                    else:
                        etaPreselect.append(True)

                #ordering data into events
                for i in range(len(ptvec)):
                    datavec = [ptvec[i], ipchi2vec[i]]

                    #uncomment to vet for fitted vertices
                    if etaPreselect[i] and truePTvec[i] > 2000 and trueTauvec[i] > 0.0002:
                    #uncomment to do no vetting
                    #if True:
                        if i <= cutoff_index:



                            try:
                                teventDict[evinseqvec[i]].append(datavec)
                                if sigtypevec[i] == 0:
                                    teventLabels[evinseqvec[i]].append(0)
                                else:
                                    teventLabels[evinseqvec[i]].append(1)

                            except:
                                teventDict[evinseqvec[i]] = [datavec]
                                if sigtypevec[i] == 0:
                                    teventLabels[evinseqvec[i]] = [0]
                                else:
                                    teventLabels[evinseqvec[i]] = [1]

                        else:

                            try:
                                eventDict[evinseqvec[i]].append(datavec)
                                if sigtypevec[i] == 0:
                                    eventLabels[evinseqvec[i]].append(0)
                                else:
                                    eventLabels[evinseqvec[i]].append(1)

                            except:
                                eventDict[evinseqvec[i]] = [datavec]
                                if sigtypevec[i] == 0:
                                    eventLabels[evinseqvec[i]] = [0]
                                else:
                                    eventLabels[evinseqvec[i]] = [1]
                    else:
                        tossed+=1
                total+=len(ptvec)
                dataMeta[f] = eventDict
                labelMeta[f] = eventLabels
                tdataMeta[f] = teventDict
                tlabelMeta[f] = teventLabels

    else:
        #two track training / testing data
        tossed=0
        total=0
        tdataMeta = {}
        tlabelMeta = {}
        dataMeta = {}
        labelMeta = {}

        for f in dataFiles:
            if f != '2018MinBias_MVATuple.root' and f != '.DS_Store':
                eventDict = {}
                eventLabels = {}
                teventDict = {}
                teventLabels = {}
                # grab the important data from each root file
                ptvec1 = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk1_PT')
                ptvec2 = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk2_PT')
                sumPTvec = np.add(ptvec1, ptvec2)
                vertexchi2vec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='sv_ENDVERTEX_CHI2')
                FDchi2vec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='sv_FDCHI2_OWNPV')


                trk1_ipchi2 = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk1_IPCHI2_OWNPV')
                trk2_ipchi2 = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk2_IPCHI2_OWNPV')

                sigtypevec1 = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk1_signal_type')
                sigtypevec2 = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk2_signal_type')
                evinseqvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='EventInSequence')

                #VETTING CODE
                mom1x = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk1_signal_TRUEP_X')
                mom1y = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk1_signal_TRUEP_Y')
                mom1z = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk1_signal_TRUEP_Z')
                mom1e = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk1_signal_TRUEP_E')
                mom2x = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk2_signal_TRUEP_X')
                mom2y = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk2_signal_TRUEP_Y')
                mom2z = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk2_signal_TRUEP_Z')
                mom2e = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk2_signal_TRUEP_E')
                truePT1 = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk1_signal_TRUEPT')
                trueTau1 = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk1_signal_TRUETAU')
                truePT2 = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk2_signal_TRUEPT')
                trueTau2 = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk2_signal_TRUETAU')

                magp1 = []
                magp2 = []
                for i in range(len(mom1x)):
                    val1 = ((mom1x[i]) ** 2 + (mom1y[i]) ** 2 + (mom1z[i]) ** 2) ** 0.5
                    val2 = ((mom2x[i]) ** 2 + (mom2y[i]) ** 2 + (mom2z[i]) ** 2) ** 0.5
                    magp2.append(val2)
                    magp1.append(val1)
                eta1 = []
                eta2 = []
                for i in range(len(magp1)):
                    val1 = math.atanh(mom1z[i] / magp1[i])
                    val2 = math.atanh(mom2z[i] / magp2[i])
                    eta1.append(val1)
                    eta2.append(val2)
                etaPreselect = []
                for i in range(len(eta1)):
                    if eta1[i] < 2 or eta1[i] > 5:
                        etaPreselect.append(False)
                    elif eta2[i] < 2 or eta2[i] > 5:
                        etaPreselect.append(False)
                    else:
                        etaPreselect.append(True)


                cutoff_index = math.floor(len(sumPTvec) * 0.75) #75% data used for training, this can be changed


                for i in range(len(sumPTvec)):
                    datavec = [sumPTvec[i], vertexchi2vec[i], FDchi2vec[i], 0]
                    if trk1_ipchi2[i] < 16: datavec[3] += 1
                    if trk2_ipchi2[i] < 16: datavec[3] += 1


                    #vetting code
                    #if True:
                    if etaPreselect[i] and truePT1[i] > 2000  and truePT2[i] > 2000 and trueTau1[i] > 0.0002 and trueTau2[i] > 0.0002:
                        if i <= cutoff_index:
                            try:
                                teventDict[evinseqvec[i]].append(datavec)
                                if sigtypevec1[i] != 0 and sigtypevec2[i] != 0:
                                    teventLabels[evinseqvec[i]].append(1)
                                else:
                                    teventLabels[evinseqvec[i]].append(0)

                            except:
                                teventDict[evinseqvec[i]] = [datavec]
                                if sigtypevec1[i] != 0 and sigtypevec2[i] != 0:
                                    teventLabels[evinseqvec[i]] = [1]
                                else:
                                    teventLabels[evinseqvec[i]] = [0]

                        else:
                            try:
                                eventDict[evinseqvec[i]].append(datavec)
                                if sigtypevec1[i] != 0 and sigtypevec2[i] != 0:
                                    eventLabels[evinseqvec[i]].append(1)
                                else:
                                    eventLabels[evinseqvec[i]].append(0)

                            except:
                                eventDict[evinseqvec[i]] = [datavec]
                                if sigtypevec1[i] != 0 and sigtypevec2[i] != 0:
                                    eventLabels[evinseqvec[i]] = [1]
                                else:
                                    eventLabels[evinseqvec[i]] = [0]
                    else:
                        tossed+=1

                total+=len(ptvec1)
                dataMeta[f] = eventDict
                labelMeta[f] = eventLabels
                tdataMeta[f] = teventDict
                tlabelMeta[f] = teventLabels
    print('tossed pct = ' + str(tossed/total))
    return tdataMeta, tlabelMeta, dataMeta, labelMeta

def testModel(model,modeldir,testData, testLabels):
    '''
    Code for testing a training BDT model, and producing and saving all
    the relevant plots to evaluating model performance
    :return: None
    '''
    modelDir=modeldir
    cutoffs = np.linspace(0, 1, 1001)

    sigdeteff_meta = np.zeros(len(cutoffs) - 1)
    trigger_meta = np.zeros(len(cutoffs) - 1)
    fpr_meta = np.zeros(len(cutoffs) - 1)
    numsig = 0
    numnoise = 0

    decays_x = []
    decays_y = []
    decays_lab = []
    for key in testData.keys():
        dsigdeteff_meta = np.zeros(len(cutoffs) - 1)
        dtrigger_meta = np.zeros(len(cutoffs) - 1)
        dfpr_meta = np.zeros(len(cutoffs) - 1)
        dnumsig = 0
        dnumnoise = 0
        print('testing for ' + str(key))

        for k in testData[key].keys():
            test = testData[key][k]
            lab = testLabels[key][k]
            noiseevent = False
            signalevent = False
            if 1 in lab:
                signalevent = True
                numsig += 1
                dnumsig += 1
            else:
                noiseevent = True
                numnoise += 1
                dnumnoise += 1
            predictionsProb = model.predict_proba(test)
            predictions = model.predict(test)

            for i in range(len(cutoffs) - 1):
                index = 0
                triggered = False
                tpfound = False
                for p in predictionsProb:
                    psig = p[1]
                    if psig >= cutoffs[i]:
                        triggered = True
                        if lab[index] == 1:
                            # signal found correctly, entire event is true positive!
                            # tp bookkeeping stuff
                            tpfound = True
                            break
                    index += 1

                if triggered and signalevent:
                    sigdeteff_meta[i] += 1
                    dsigdeteff_meta[i] += 1
                elif triggered and noiseevent:
                    fpr_meta[i] += 1
                    dfpr_meta[i] += 1

        dsigdeteff_meta = np.divide(dsigdeteff_meta, dnumsig)
        dfpr_meta = np.divide(dfpr_meta, dnumnoise)
        decays_x.append(dfpr_meta)
        decays_y.append(dsigdeteff_meta)
        decays_lab.append(key.split('.')[0].split('_')[0])

    sigdeteff_meta = np.divide(sigdeteff_meta, numsig)
    fpr_meta = np.divide(fpr_meta, numnoise)
    fpr_meta = np.multiply(fpr_meta, 30000)
    fpr_meta = np.log10(fpr_meta)

    # make decay roc plots
    decays_x = np.multiply(decays_x, 30000)
    newdecays_x = []
    for x in decays_x:
        newdecays_x.append(np.log10(x))
    decays_x = newdecays_x
    plt.figure()
    lab0, = plt.step(decays_x[0], decays_y[0], label=decays_lab[0])
    lab1, = plt.step(decays_x[1], decays_y[1], label=decays_lab[1])
    lab2, = plt.step(decays_x[2], decays_y[2], label=decays_lab[2])
    lab3, = plt.step(decays_x[3], decays_y[3], label=decays_lab[3])
    lab4, = plt.step(decays_x[4], decays_y[4], label=decays_lab[4])
    lab5, = plt.step(decays_x[5], decays_y[5], label=decays_lab[5])
    plt.xlabel('Trigger Rate (Log KHz)')
    plt.ylabel('Signal Detection Efficiency')
    plt.legend(handles=[lab0, lab1, lab2, lab3, lab4, lab5])
    plt.title('Classifier Performance by Interesting Decay Type')
    plt.savefig(modelDir + '/decayRoc.pdf')
    with open(modeldir + '/decayperformance/logfpr.txt', 'w') as f:
        json.dump(str(decays_x), f)
    with open(modeldir + '/decayperformance/sigdet.txt', 'w') as f:
        json.dump(str(decays_y), f)
    with open(modeldir + '/decayperformance/labels.txt', 'w') as f:
        json.dump(str(decays_lab), f)

    # make total roc plot
    plt.figure()
    plt.step(fpr_meta, sigdeteff_meta)
    plt.xlabel('Trigger Rate (Log KHz)')
    plt.ylabel('Signal Detection Efficiency')
    plt.title('Detection Efficiency vs Trigger Rate')
    plt.savefig(modelDir + '/totalRoc.pdf')

    with open(modeldir + '/cutoffs.txt', 'w') as f:
        json.dump(list(cutoffs), f)
    with open(modeldir + '/eff.txt', 'w') as f:
        json.dump(list(sigdeteff_meta), f)
    with open(modeldir + '/trigger.txt', 'w') as f:
        json.dump(list(fpr_meta), f)



if __name__=='__main__':
    #parse command line arguments
    args = parseArgs()
    params =vars(args)
    if params['dataDir'] == 'AllenMVA/data':
        cdw = os.getcwd()
        cdwSplit= cdw.split('/')
        cdw = ''
        for c in cdwSplit[1:-1]:
            cdw+='/'
            cdw+=c
        cdw+='/'
        dataDir = cdw+'AllenMVA/data/'
    else:
        dataDir = params['dataDir']
    if params['modelType'] == 'oneTrack':
        oneTrack = True
    elif params['modelType']== 'twoTrack':
        oneTrack = False
    else:
        oneTrack = '' #raises appropriate error

    #load training data
    print('loading training and testing data ...')
    trainData, trainLabels, testData, testLabels = createData(oneTrack,dataDir)

    #reformat training data into a list
    trainingDataList = []
    trainingLabelList = []

    testingDataList = []
    testingLabelList = []
    printed=0

    #learn about decay type, event, and example frequencies for weighting
    #we can only weight by decay type for specific classifiers
    decayTypeFrequency={}
    eventTypeFrequency = {'noise':0,'signal':0}
    for key in trainData.keys():
        decayTypeFrequency[key] = 0
        for k in trainData[key].keys():
            for i in trainData[key][k]:
                trainingDataList.append(i)
                decayTypeFrequency[key]+=1
                if printed <=10:
                    #print(i)
                    printed+=1
            for l in trainLabels[key][k]:
                trainingLabelList.append(l)
            if 1 in trainLabels[key][k]:
                eventTypeFrequency['signal']+=1
            else:
                eventTypeFrequency['noise']+=1
    for key in testData.keys():
        for k in testData[key].keys():
            for i in testData[key][k]:
                testingDataList.append(i)
            for l in testLabels[key][k]:
                testingLabelList.append(l)

    print('Training and testing data loaded successfully!')

    #train the model
    print('training model ...')
    numnoise = 0
    numsig = 0
    for i in trainingLabelList:
        if i == 0:
            numnoise += 1
        else:
            numsig += 1

    #choose the noise vs signal weighting method
    if params['weightingType'] == 'example':
        class_weights = [1, numnoise / numsig]
    elif params['weightingType'] == 'event':
        class_weights = [1, eventTypeFrequency['noise']/eventTypeFrequency['signal']]
    else:
        raise customException('Please input a valid argument for --weightingType')

    custom_metric = ['Accuracy', 'BalancedAccuracy']
    modeldir = params['modelDir']
    if modeldir == './catBoostClassifiers/':
        cdw2 = os.getcwd()
        cdwSplit2 = cdw.split('/')
        cdw2 = ''
        for c in cdwSplit2[1:-1]:
            cdw2 += '/'
            cdw2 += c
        cdw2 += '/'
        modeldir = cdw + 'catBoostClassifiers/custom/'+params['modelType']+'_generic'+'_VET'+'_'+params['weightingType']+'_'+params['randomShuffle']
    try:
        os.listdir(modeldir)
    except:
        print('model directory doesnt exist, making directory at '+modeldir)
        os.mkdir(modeldir)
    model = CatBoostClassifier(iterations=int(params['iterations']),
                               depth=int(params['depth']),
                               learning_rate=float(params['learningRate']),
                               loss_function='Logloss',
                               train_dir=modeldir,
                               class_weights=class_weights,
                               custom_metric=custom_metric,
                               verbose=False
                               )

    #randomly shuffle training data / labels if necessary
    if params['randomShuffle'] == 'True':
        random.seed(4)
        random.shuffle(trainingDataList)
        random.seed(4)
        random.shuffle(trainingLabelList)
    elif params['randomShuffle'] == 'False':
        pass
    else:
        raise customException('Please input a valid argument for --randomShuffle')
    model.fit(trainingDataList, trainingLabelList)
    model.save_model(modeldir+'/model.cbm')

    trainFileDir = modeldir+'/learn_error.tsv'
    logloss = []
    iter = []
    balacc = []
    with open(trainFileDir,'r') as f:
        content = f.readlines()
        for line in content[1:]:
            linesp = line.split('\t')
            iter.append(int(linesp[0]))
            logloss.append(float(linesp[1]))
            balacc.append(float(linesp[5]))
    plt.figure()
    losslab, = plt.step(iter,logloss,label='Log Loss')
    balacclab, = plt.step(iter,balacc,label='Balanced Accuracy')
    plt.title('Performance Metrics throughout Training')
    plt.ylabel('Metric Value')
    plt.xlabel('iteration')
    plt.legend(handles=[losslab,balacclab])
    plt.savefig(modeldir+'/training.pdf')
    print('model trained successfully!')

    #model testing
    print('testing model ...')
    os.mkdir(modeldir+'/decayperformance')

    testModel(model=model,modeldir=modeldir,testData=testData,testLabels=testLabels)


