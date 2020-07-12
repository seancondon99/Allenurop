import sys, os
import numpy as np
import matplotlib.pyplot as plt
import uproot
import math
from catboost import CatBoostClassifier, Pool
import json
import random
import uproot_custom

tree_dict = {
    'N1Trk' : 'DecayTreeTuple/N1Trk',
    'N2Trk' : 'DecayTreeTuple#1/N2Trk',
    'N3Trk' : 'DecayTreeTuple#2/N3Trk',
    'N4Trk' : 'DecayTreeTuple#3/N4Trk'
}


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
            if f == '2018MinBias_MVATuple.root':
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
            if f == '2018MinBias_MVATuple.root':
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
                        if True:
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
                        tossed+=1

                total+=len(ptvec1)
                tdataMeta[f] = teventDict
                tlabelMeta[f] = teventLabels
    print('tossed pct = ' + str(tossed/total))
    return tdataMeta,tlabelMeta

def cutbasedcreateData(oneTrack,dataDir):
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
            if f == '2018MinBias_MVATuple.root':
                eventDict = {}
                eventLabels = {}


                ptvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_PT')
                ipchi2vec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_IPCHI2_OWNPV')
                chi2vec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_OWNPV_CHI2')

                ndofvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_OWNPV_NDOF')
                sigtypevec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='trk_signal_type')
                evinseqvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[0], var='EventInSequence')

                # VET THE PARENT PARTICLES SO WE KNOW THEY COULD BE PROPERLY RECONSTRUCTED
                # vetting code...
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
            if f == '2018MinBias_MVATuple.root':
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

                # VETTING CODE
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
                mcorvec = uproot_custom.findData(dir=allenMVAdir + f, tree=trunks[1], var='sv_MCORR')

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

                cutoff_index = math.floor(len(sumPTvec) * 0.75)

                # preselection data
                pt1Preselect = []
                pt2Preselect = []
                ip1Preselect = []
                ip2Preselect = []
                mcorPreselect = []
                for i in range(len(ptvec1)):

                    if ptvec1[i] > 500:
                        pt1Preselect.append(True)
                    else:
                        pt1Preselect.append(False)

                    if ptvec2[i] > 500:
                        pt2Preselect.append(True)
                    else:
                        pt2Preselect.append(False)

                    if trk1_ipchi2[i] > 4:
                        ip1Preselect.append(True)
                    else:
                        ip1Preselect.append(False)

                    if trk2_ipchi2[i] > 4:
                        ip2Preselect.append(True)
                    else:
                        ip2Preselect.append(False)

                    if mcorvec[i] > 1:
                        mcorPreselect.append(True)
                    else:
                        mcorPreselect.append(False)



                for i in range(len(sumPTvec)):
                    datavec = [ptvec1[i],ptvec2[i],sumPTvec[i],0, trk1_ipchi2[i],trk2_ipchi2[i],mcorrvec[i], eta[i],vertexchi2vec[i]]
                    if trk1_ipchi2[i] < 16: datavec[3] += 1
                    if trk2_ipchi2[i] < 16: datavec[3] += 1



                    #vetting code
                    #if True:
                    if etaPreselect[i] and truePT1[i] > 2000  and truePT2[i] > 2000 and trueTau1[i] > 0.0002 and trueTau2[i] > 0.0002:

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

    print('tossed pct = ' + str(tossed/total))

    return dataMeta, labelMeta


def oneTrackCut(data,label):
    #datavec = [ptvec[i], ipchi2vec[i]]
    triggered = False
    signal = False
    for i in range(len(data)):
        v = data[i]
        l = label[i]
        if l == 1: signal = True
        pt=v[0]
        ipchi2 = v[1]
        if ipchi2 < 0:
            return False, False
        if True:
            if pt > 2000:
                if pt < 26000:
                    #complicated expression
                    val1 = math.log(ipchi2)
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



def triggerRateMeasurement(oneTrack, modeldir):
    '''
    This function is intended to measure the trigger rate of a model specified by the modeldir parameter
    by counting the number of events triggered on in the 2018MinBias_MVATuple.root file. The function
    then compares this trigger rate to the cut-based classifier's trigger rate.
    :param oneTrack: Bool, true if using a one-track model, false if using a two-track model
    :param modeldir: Str, directory pointing to the .cbm file that contains the BDT model weights
    :return: None
    '''

    data, labels = createData(oneTrack=oneTrack,dataDir='/Users/seancondon/Desktop/LHC_Urop/AllenMVA/data/')
    testData, testLabels = data,labels
    print('trigger rate data loaded.')
    for key in data.keys():
        pass
    model = CatBoostClassifier()
    model = model.load_model(modeldir)

    #TEST TRIGGER RATE
    cutoffs = np.linspace(0, 1, 1001)

    sigdeteff_meta = np.zeros(len(cutoffs) - 1)
    trigger_meta = np.zeros(len(cutoffs) - 1)
    fpr_meta = np.zeros(len(cutoffs) - 1)
    trigger_meta = np.zeros(len(cutoffs) - 1)
    numsig = 0
    numnoise = 0

    completed = 0
    for key in testData.keys():
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

                if triggered:
                    trigger_meta[i]+=1

            completed+=1
            if completed%1000 == 0:
                print(str((completed/len(testData[key].keys() ) *100))[:4] + '% done')

    print('BDT RESULTS!!!')
    print(numsig,numnoise)
    trigger_meta = np.divide(trigger_meta, numnoise+numsig)
    trigger_meta = np.multiply(trigger_meta, 30000)
    trigger_meta = np.log10(trigger_meta)

    for i in range(len(cutoffs)):
        if trigger_meta[i] < 3:
            print('Cutoff probability of ' + str(cutoffs[i])+ ' yields a trigger rate of about 1 MHz')
            break


    print('CUT BASED RESULTS!!!')
    data,labels=cutbasedcreateData(oneTrack,dataDir='/Users/seancondon/Desktop/LHC_Urop/AllenMVA/data/')
    #OPTIONAL, compare the cutbased classifier
    triggeredtot = 0
    tot = 0
    ptvals = []
    ipchi2vals = []

    for key in data.keys():
        for k in data[key].keys():
            if oneTrack:
                trig, sig = oneTrackCut(data=data[key][k], label=labels[key][k])
                for i in data[key][k]:
                    ptvals.append(i[0])
                    ipchi2vals.append(i[1])
            elif not oneTrack:
                trig, sig = twoTrackCut(data=data[key][k], label=labels[key][k])

            tot += 1
            if trig:
                triggeredtot += 1

    print('trigger rate = ' +str(triggeredtot/tot))




triggerRateMeasurement(True, modeldir='/Users/seancondon/Desktop/LHC_urop/catBoostClassifiers/vetting/oneTrack_generic_VET/model.cbm')
