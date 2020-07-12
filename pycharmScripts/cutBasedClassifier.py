import sys, os
import numpy as np
import matplotlib.pyplot as plt
import uproot
import math
import json

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
    raise customException('Make sure trainClassifier.py is in the same directory as uproot_custom.py')


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

        dataMeta = {}
        labelMeta = {}

        for f in dataFiles:
            if f != '2018MinBias_MVATuple.root' and f != '.DS_Store':
                eventDict = {}
                eventLabels = {}


                ptvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_PT')
                ipchi2vec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_IPCHI2_OWNPV')
                chi2vec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_OWNPV_CHI2')

                ndofvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_OWNPV_NDOF')
                sigtypevec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_signal_type')
                evinseqvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='EventInSequence')


                #ordering data into events
                for i in range(len(ptvec)):
                    datavec = [ptvec[i], ipchi2vec[i], chi2vec[i],ndofvec[i]]

                    #uncomment to vet for fitted vertices
                    #if etaPreselect[i] and truePTvec[i] > 2000 and trueTauvec[i] > 0.0002:
                    #uncomment to do no vetting
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
                        tossed+=1

                total+=len(ptvec)
                dataMeta[f] = eventDict
                labelMeta[f] = eventLabels


    else:
        #two track training / testing data
        tossed=0
        total=0
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
                mcorrvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='sv_MCORR')


                trk1_ipchi2 = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk1_IPCHI2_OWNPV')
                trk2_ipchi2 = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk2_IPCHI2_OWNPV')

                sigtypevec1 = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk1_signal_type')
                sigtypevec2 = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='trk2_signal_type')
                evinseqvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='EventInSequence')

                momx = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='sv_PX')
                momy = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='sv_PY')
                momz = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='sv_PZ')
                mome = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='sv_PE')

                magp = []
                for i in range(len(momx)):
                    val = ((momx[i]) ** 2 + (momy[i]) ** 2 + (momz[i]) ** 2) ** 0.5
                    magp.append(val)
                eta = []
                for i in range(len(magp)):
                    val = math.atanh(momz[i] / magp[i])
                    eta.append(val)



                for i in range(len(sumPTvec)):
                    datavec = [ptvec1[i],ptvec2[i],sumPTvec[i],0, trk1_ipchi2[i],trk2_ipchi2[i],mcorrvec[i], eta[i],vertexchi2vec[i]]
                    if trk1_ipchi2[i] < 16: datavec[3] += 1
                    if trk2_ipchi2[i] < 16: datavec[3] += 1



                    #vetting code
                    if True:
                    #if etaPreselect[i] and truePT1[i] > 2000  and truePT2[i] > 2000 and trueTau1[i] > 0.0002 and trueTau2[i] > 0.0002:

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

    #print('tossed pct = ' + str(tossed/total))

    return dataMeta, labelMeta

def oneTrackCut(data,label):
    #datavec = [ptvec[i], ipchi2vec[i], chi2vec[i],ndofvec[i]]
    triggered = False
    signal = False
    for i in range(len(data)):
        v = data[i]
        l = label[i]
        if l == 1: signal = True
        pt=v[0]
        ipchi2 = v[1]
        chi2 = v[2]
        ndof = v[3]
        #if chi2/ndof < 2.5:
        if True:
            if pt > 2000:
                if pt < 26000:
                    #complicated expression
                    try:
                        val1 = math.log(ipchi2)
                    except:
                        val1=-100000
                    val2 = (1/(pt/1000 -1)**2) + (1.248/(26*(26-pt/1000)))+math.log(7.4)
                    if val1 > val2:
                        triggered = True
                elif pt >= 26000:
                    #less complicated expression
                    if ipchi2 > 7.4:
                        triggered = True


    return triggered,signal

def twoTrackCut(data,label):
    #datavec = [ptvec1[i],ptvec2[i],sumPTvec[i],0, trk1_ipchi2[i],trk2_ipchi2[i],mcorrvec[i], eta[i],vertexchi2vec[i]]
    triggered = False
    signal = False
    for i in range(len(data)):
        v = data[i]
        l = label[i]
        if l ==1:
            signal=True
        pt1 = v[0]
        pt2 = v[1]
        sumpt = v[2]
        n = v[3]
        ipchi2_1 = v[4]
        ipchi2_2 = v[5]
        mcorr = v[6]
        eta = v[7]
        vertexchi2 = v[8]
        if vertexchi2 < 25:
            if min(pt1,pt2)>700:
                if min(ipchi2_1,ipchi2_2) > 12:
                    if sumpt > 2000:
                        if mcorr > 1000:
                            if eta > 2 and eta < 5:
                                if n < 1:
                                    triggered = True
        return triggered,signal


if __name__=='__main__':
    print('running!')
    data,labels = createData(oneTrack=False,dataDir='/Users/seancondon/Desktop/LHC_urop/AllenMVA/data/')
    print('Data loaded correctly')
    triggeredtot = 0
    tot = 0
    signaltotal = 0
    noisetotal=0

    sigdet=0
    fpr=0

    for key in data.keys():
        for k in data[key].keys():
            trig, sig = twoTrackCut(data=data[key][k],label=labels[key][k])
            tot+=len(labels[key][k])
            if trig:
                triggeredtot+=1
            if sig:
                signaltotal+=1
            if not sig:
                noisetotal+=1
            if trig and sig:
                sigdet+=1
            if trig and not sig:
                fpr+=1
    print(signaltotal)
    print(noisetotal)
    print('\n')
    print(triggeredtot)
    print(sigdet)
    print(fpr)
    print('signal detection eff')
    print(sigdet/signaltotal)
    print('false positive rate')
    print(fpr/noisetotal)




