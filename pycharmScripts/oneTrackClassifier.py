import numpy as np
from catboost import CatBoostClassifier, Pool
import math
import uproot_custom
import os
import matplotlib.pyplot as plt
import json
import time


allenMVAdir = '/Users/seancondon/Desktop/LHC_urop/AllenMVA/data/'
tupleTrees = ['2018MinBias_MVATuple.root',
              'Bs2JPsiPhiMD_MVATuple.root',
              'BsPhiPhiMD_MVATuple.root',
              'Ds2KKPiMD_MVATuple.root',
              'Dst2D0piMD_MVATuple.root',
              'KstEEMD_MVATuple.root',
              'KstMuMuMD_MVATuple.root']

trunks = ['N1Trk', 'N2Trk', 'N3Trk', 'N4Trk']

tree_dict = {
    'N1Trk' : 'DecayTreeTuple/N1Trk',
    'N2Trk' : 'DecayTreeTuple#1/N2Trk',
    'N3Trk' : 'DecayTreeTuple#2/N3Trk',
    'N4Trk' : 'DecayTreeTuple#3/N4Trk'
}

def create_data():
    '''
    Function to pool all the training and testing data needed for the 1-track
    classifier from the six .root files specified in tupleTrees
    :return: 4 ndarrays containing the training data, training labels
    testing data, and testing labels
    '''
    trainmeta = []
    sigtype_train = []

    testmeta = []
    sigtype_test = []

    analysis_variables = ['trk_PT','trk_IPCHI2_OWNPV','trk_signal_type']
    for f in tupleTrees:
        if f != '2018MinBias_MVATuple.root':
            #grab the important data from each root file
            ptvec = uproot_custom.findData(dir=allenMVAdir+f,tree=trunks[0],var='trk_PT')
            ipchi2vec = uproot_custom.findData(dir=allenMVAdir+f,tree=trunks[0],var='trk_IPCHI2_OWNPV')
            sigtypevec = uproot_custom.findData(dir=allenMVAdir+f,tree=trunks[0],var='trk_signal_type')
            cutoff_index = math.floor(len(ptvec)*0.75)

            for i in range(len(ptvec)):
                datavec = [ptvec[i],ipchi2vec[i]]

                if i <= cutoff_index:
                    trainmeta.append(datavec)
                    if sigtypevec[i] == 0:
                        sigtype_train.append(0)
                    else:
                        sigtype_train.append(1)

                else:
                    testmeta.append(datavec)
                    if sigtypevec[i] == 0:
                        sigtype_test.append(0)
                    else:
                        sigtype_test.append(1)

    return trainmeta, sigtype_train, testmeta, sigtype_test


def train_model(traindata, trainlabels,modeldir,learning_rate=0.03,
                iterations=1000, depth=10,verbose=False):
    '''
    Trains a CatBoostClassifier with the specified modeldir, learning_rate,
    iterations, and depth using the training data and labels provided.
    :return: trained catboostclassifier model
    '''
    numnoise = 0
    numsig = 0
    for i in trainlabels:
        if i == 0:
            numnoise+=1
        else:
            numsig+=1

    class_weights = [1, numnoise/numsig]

    custom_metric = ['Accuracy', 'BalancedAccuracy']
    #these datasets are imbalanced so we have to modify weights with the
    #class_weights parameter
    model = CatBoostClassifier(iterations=iterations,
                               depth=depth,
                               learning_rate=learning_rate,
                               loss_function='Logloss',
                               train_dir = modeldir,
                               class_weights=class_weights,
                               custom_metric=custom_metric,
                               verbose=verbose
                               )
    # train the model
    model.fit(traindata, trainlabels)

    return model

def test_model(model, modeldir, testdata, testlabels, r):
    '''
    Evaluates a trained CatBoostClassifier on the provided testing data / labels.
    :return: None, ROC curves saved in directory specified below
    '''

    predictions = model.predict_proba(testdata)
    cutoffs = np.linspace(0,1,101)

    #for roc we need signal det eff and fpr as function of cutoff
    sigdeteff_meta = np.zeros(len(cutoffs)-1)
    fpr_meta = np.zeros(len(cutoffs)-1)
    index = 0
    for p in predictions:
        psig = p[1]
        for i in range(len(cutoffs)-1):
            if psig >= cutoffs[i]:
                #signal detected!
                if testlabels[index] == 0:
                    #false positive
                    fpr_meta[i] += 1
                else:
                    #true positive
                    sigdeteff_meta[i] +=1
        index+=1

    nsig = 0
    nnoise = 0
    for l in testlabels:
        if l == 0: nnoise+=1
        else: nsig +=1

    norm_eff = np.divide(sigdeteff_meta,nsig)
    norm_fpr = np.divide(fpr_meta,nnoise)
    plt.figure()
    plt.step(norm_fpr,norm_eff)
    plt.xlabel('False Positive Rate')
    plt.ylabel('Signal Detection Efficiency')
    plt.title('Classifier ROC for learning rate = ' + str(r))
    plt.savefig(modeldir + '/roc.pdf')

    #also dump the cutoff, eff, and fpr data
    with open(modeldir+'/cutoffs.txt','w') as f:
        json.dump(list(cutoffs),f)
    with open(modeldir+'/eff.txt','w') as f:
        json.dump(list(norm_eff),f)
    with open(modeldir+'/fpr.txt','w') as f:
        json.dump(list(norm_fpr),f)




def vary_everthing():
    '''
    Varies the learning rate, depth, and iterations in a 3D parameter space
    defined below. Used to optimize hyperparameters.
    :return: None
    '''
    learningrates = np.logspace(-1, -3, 3)
    depths = [6,8,10,12,14,16]
    iterations_tries = [100,200,400,800]

    traindir_prelim = '/Users/seancondon/Desktop/LHC_urop/catBoostClassifiers/oneTrack/allVariation/'

    traindata, trainlabels, testdata, testlabels = create_data()
    print('training / testing data loaded')

    modeln = 0
    for l in learningrates:
        for d in depths:
            for i in iterations_tries:
                print('beginning model number ' + str(modeln))
                t0 = time.time()
                traindir = traindir_prelim + str(l)+'_'+str(d)+'_'+str(i)
                try:
                    os.mkdir(traindir)
                except:
                    pass

                model = train_model(traindata=traindata,
                                    trainlabels=trainlabels,
                                    modeldir=traindir,
                                    learning_rate=l,
                                    depth=d,
                                    iterations=i)

                # test model for roc
                test_model(model, traindir, testdata, testlabels, l)
                t1 = time.time()

                print('this model took ' + str(t1 - t0) + ' seconds')
                modeln += 1


def trainOptimal():

    traindata, trainlabels, testdata, testlabels = create_data()
    print('training / testing data loaded')

    traindir = '/Users/seancondon/Desktop/LHC_urop/catBoostClassifiers/oneTrack/optimal/'
    model = train_model(traindata=traindata,
                        trainlabels=trainlabels,
                        modeldir=traindir,
                        learning_rate=0.01,
                        depth=8,
                        iterations=800,
                        verbose=True)
    model.save_model(fname=traindir + 'savedModel')

#trainOptimal()

def eventTestData():
    #want like [[event1 data], [event2 data], ... [eventn data]]
    tdataMeta = {}
    tlabelMeta = {}
    dataMeta = {}
    labelMeta = {}

    tottossed = 0
    totot = 0


    analysis_variables = ['trk_PT', 'trk_IPCHI2_OWNPV', 'trk_signal_type']
    for f in tupleTrees:
        if f != '2018MinBias_MVATuple.root':
            eventDict = {}
            eventLabels = {}
            teventDict = {}
            teventLabels = {}
            # grab the important data from each root file
            ptvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_PT')
            ipchi2vec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_IPCHI2_OWNPV')
            sigtypevec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_signal_type')
            evinseqvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='EventInSequence')
            cutoff_index = math.floor(len(ptvec) * 0.75)

            #data for doing preselections
            ptpreselect = []
            for i in ptvec:
                if i > 500:
                    ptpreselect.append(True)
                else:
                    ptpreselect.append(False)
            ipchi2preselect= []
            for i in ipchi2vec:
                if i > 4:
                    ipchi2preselect.append(True)
                else:
                    ipchi2preselect.append(False)

            tossed = 0
            for i in range(len(ptvec)):
                datavec = [ptvec[i], ipchi2vec[i]]

                if i <= cutoff_index:
                    #if ptpreselect[i] and ipchi2preselect[i]:
                    if True:
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
                        tossed+=1

                else:
                    #if ptpreselect[i] and ipchi2preselect[i]:
                    if True:
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
                        tossed +=1
            tottossed+=tossed
            totot += len(ptvec)
            dataMeta[f] = eventDict
            labelMeta[f] = eventLabels
            tdataMeta[f] = teventDict
            tlabelMeta[f] = teventLabels



    return dataMeta, labelMeta, tdataMeta, tlabelMeta





def testOptimal():

    modelDir = '/Users/seancondon/Desktop/LHC_urop/catBoostClassifiers/oneTrack/optimal/savedModel'

    #model = CatBoostClassifier(iterations=800,
    #                           depth=8,
    #                           learning_rate=0.01,
    #                           loss_function='Logloss',
    #                           train_dir = modelDir
    #                           )
    #model = model.load_model(fname=modelDir)

    testData, testLabels, trainData, trainLabels = eventTestData()
    traind = []
    trainl = []
    for key in trainData.keys():
        for k in trainData[key].keys():
            for i in trainData[key][k]:
                traind.append(i)
            for l in trainLabels[key][k]:
                trainl.append(l)

    print('testing data loaded.')

    model = train_model(traindata=traind,trainlabels=trainl,modeldir='/Users/seancondon/Desktop/LHC_urop/catBoostClassifiers/oneTrack/optimalTesting',
                        learning_rate=0.01,
                        iterations=100, depth=8,verbose=True)

    cutoffs = np.linspace(0, 1, 101)
    lablengths = []

    sigdeteff_meta = np.zeros(len(cutoffs) - 1)
    trigger_meta = np.zeros(len(cutoffs) - 1)
    fpr_meta = np.zeros(len(cutoffs) - 1)
    numsig = 0
    numnoise = 0
    for key in testData.keys():
        print('testing for '+ str(key))

        for k in testData[key].keys():
            test = testData[key][k]
            lab = testLabels[key][k]
            lablengths.append(len(lab))
            noiseevent = False
            signalevent = False
            if 1 in lab:
                signalevent = True
                numsig += 1
            else:
                noiseevent = True
                numnoise += 1
            predictionsProb = model.predict_proba(test)
            predictions = model.predict(test)


            for i in range(len(cutoffs)-1):
                index = 0
                triggered = False
                tpfound = False
                for p in predictionsProb:
                    psig = p[1]
                    if psig >= cutoffs[i]:
                        triggered = True
                        if lab[index] == 1:
                            #signal found correctly, entire event is true positive!
                            #tp bookkeeping stuff
                            tpfound = True
                            break
                    index+=1

                if triggered and signalevent:
                    sigdeteff_meta[i] += 1
                if triggered and noiseevent:
                    fpr_meta[i] += 1

    sigdeteff_meta = np.divide(sigdeteff_meta,numsig)
    fpr_meta = np.divide(fpr_meta,numnoise)
    plt.figure()
    plt.step(fpr_meta,sigdeteff_meta)
    plt.xlabel('false positive rates')
    plt.ylabel('signal det eff')
    plt.savefig('/Users/seancondon/Desktop/LHC_urop/catBoostClassifiers/oneTrack/optimalTesting/' + 'totalRoc.pdf')

def overtrainingVerification(trainData,trainLabels,model):

    testData,testLabels = trainData,trainLabels
    cutoffs = np.linspace(0, 1, 1001)
    sigdeteff_meta = np.zeros(len(cutoffs) - 1)
    trigger_meta = np.zeros(len(cutoffs) - 1)
    fpr_meta = np.zeros(len(cutoffs) - 1)
    numsig = 0
    numnoise = 0

    for key in trainData.keys():
        print('testing train for '+ str(key))

        for k in testData[key].keys():
            test = testData[key][k]
            lab = testLabels[key][k]
            noiseevent = False
            signalevent = False
            if 1 in lab:
                signalevent = True
                numsig += 1
            else:
                noiseevent = True
                numnoise += 1
            predictionsProb = model.predict_proba(test)
            predictions = model.predict(test)


            for i in range(len(cutoffs)-1):
                index = 0
                triggered = False
                tpfound = False
                for p in predictionsProb:
                    psig = p[1]
                    if psig >= cutoffs[i]:
                        triggered = True
                        if lab[index] == 1:
                            #signal found correctly, entire event is true positive!
                            #tp bookkeeping stuff
                            tpfound = True
                            break
                    index+=1

                if triggered and signalevent:
                    sigdeteff_meta[i] += 1
                if triggered and noiseevent:
                    fpr_meta[i] += 1

    sigdeteff_meta = np.divide(sigdeteff_meta,numsig)
    fpr_meta = np.divide(fpr_meta,numnoise)
    fpr_meta = np.multiply(fpr_meta,30000)
    return sigdeteff_meta,fpr_meta

def testOptimalFull():

    tossedpct=0

    modelDir = '/Users/seancondon/Desktop/LHC_urop/catBoostClassifiers/oneTrack/preselections/none'
    try:
        os.listdir(modelDir)
    except:
        os.mkdir(modelDir)
    modeldir = modelDir
    try:
        os.mkdir(modelDir+'/decayperformance')
    except:
        pass

    testData, testLabels, trainData, trainLabels = eventTestData()
    traind = []
    trainl = []
    for key in trainData.keys():
        for k in trainData[key].keys():
            for i in trainData[key][k]:
                traind.append(i)
            for l in trainLabels[key][k]:
                trainl.append(l)

    print('testing data loaded.')
    print(traind[:10])
    print(trainl[:10])

    model = train_model(traindata=traind,trainlabels=trainl,modeldir=modelDir,
                        learning_rate=0.01,
                        iterations=800, depth=8)

    cutoffs = np.linspace(0, 1, 1001)
    lablengths = []

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
        print('testing for '+ str(key))

        for k in testData[key].keys():
            test = testData[key][k]
            lab = testLabels[key][k]
            lablengths.append(len(lab))
            noiseevent = False
            signalevent = False
            if 1 in lab:
                signalevent = True
                numsig += 1
                dnumsig+=1
            else:
                noiseevent = True
                numnoise += 1
                dnumnoise+=1
            predictionsProb = model.predict_proba(test)
            predictions = model.predict(test)


            for i in range(len(cutoffs)-1):
                index = 0
                triggered = False
                tpfound = False
                for p in predictionsProb:
                    psig = p[1]
                    if psig >= cutoffs[i]:
                        triggered = True
                        if lab[index] == 1:
                            #signal found correctly, entire event is true positive!
                            #tp bookkeeping stuff
                            tpfound = True
                            break
                    index+=1

                if triggered and signalevent:
                    sigdeteff_meta[i] += 1
                    dsigdeteff_meta[i] += 1
                if triggered and noiseevent:
                    fpr_meta[i] += 1
                    dfpr_meta[i] += 1

        dsigdeteff_meta = np.divide(dsigdeteff_meta,dnumsig)
        dfpr_meta = np.divide(dfpr_meta, dnumnoise)
        decays_x.append(dfpr_meta)
        decays_y.append(dsigdeteff_meta)
        decays_lab.append(key.split('.')[0].split('_')[0])

    sigdeteff_meta = np.divide(sigdeteff_meta,numsig)
    fpr_meta = np.divide(fpr_meta,numnoise)
    fpr_meta = np.multiply(fpr_meta,30000)
    fpr_meta = np.log10(fpr_meta)

    #test training data for overtraining

    #make decay roc plots
    decays_x = np.multiply(decays_x,30000)
    newdecays_x = []
    for x in decays_x:
        newdecays_x.append(np.log10(x))
    decays_x = newdecays_x
    plt.figure()
    lab0, = plt.step(decays_x[0],decays_y[0], label=decays_lab[0])
    lab1, = plt.step(decays_x[1], decays_y[1], label=decays_lab[1])
    lab2, = plt.step(decays_x[2], decays_y[2], label=decays_lab[2])
    lab3, = plt.step(decays_x[3], decays_y[3], label=decays_lab[3])
    lab4, = plt.step(decays_x[4], decays_y[4], label=decays_lab[4])
    lab5, = plt.step(decays_x[5], decays_y[5], label=decays_lab[5])
    plt.xlabel('Trigger Rate (Log KHz)')
    plt.ylabel('Signal Detection Efficiency')
    plt.legend(handles=[lab0,lab1,lab2,lab3,lab4,lab5])
    plt.title('Classifier Performance by Interesting Decay Type')
    plt.savefig(modelDir+'/decayRoc.pdf')
    with open(modeldir+'/decayperformance/logfpr.txt','w') as f:
        json.dump(str(decays_x),f)
    with open(modeldir+'/decayperformance/sigdet.txt','w') as f:
        json.dump(str(decays_y),f)
    with open(modeldir+'/decayperformance/labels.txt','w') as f:
        json.dump(str(decays_lab),f)



    #make total roc plot
    plt.figure()
    plt.step(fpr_meta,sigdeteff_meta)
    plt.xlabel('Trigger Rate (Log KHz)')
    plt.ylabel('Signal Detection Efficiency')
    plt.title('Detection Efficiency vs Trigger Rate')
    plt.savefig(modelDir+'/totalRoc.pdf')

    with open(modeldir+'/cutoffs.txt','w') as f:
        json.dump(list(cutoffs),f)
    with open(modeldir+'/eff.txt','w') as f:
        json.dump(list(sigdeteff_meta),f)
    with open(modeldir+'/trigger.txt','w') as f:
        json.dump(list(fpr_meta),f)
    with open(modeldir+'/tossed.txt','w') as f:
        json.dump(str(tossedpct),f)

    #make overtraining plot
    train_sigdet, train_fpr = overtrainingVerification(trainData=trainData,trainLabels=trainLabels,model=model)
    train_fpr = np.log10(train_fpr)
    plt.figure()
    labtest, = plt.step(fpr_meta,sigdeteff_meta,label='Testing Data Performance')
    labtrain, = plt.step(train_fpr,train_sigdet,label='Training Data Performance')
    plt.xlabel('Trigger Rate (Log KHz)')
    plt.ylabel('Signal Detection Efficiency')
    plt.title('Comparative Performance on Testing vs. Training Data')
    plt.legend(handles=[labtest,labtrain])
    plt.savefig(modelDir+'/overtrainRoc.pdf')

testOptimalFull()













