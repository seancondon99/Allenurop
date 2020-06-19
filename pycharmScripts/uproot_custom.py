import sys, os
import numpy as np
import uproot


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

def findData(dir, tree, var):
    '''
    Locates the data in the .root file specified by dir and tree,
    returns a numpy array containing that data
    '''
    #uproot.open returns the root file specified by dir
    file = uproot.open(dir)
    #get the correct file for the ttree specified by tree argument
    treename = tree_dict[tree]
    #we can return this tree simply by indexing the file object
    ttree = file[treename]
    #finally we can turn this ttree into a pandas dataframe
    ttree_data = ttree.pandas.df()

    columnlist = ttree_data.columns

    try:
        return ttree_data[var].values
    except:
        print('Variable does not exist in this tree!')
        return None


def exploreBranch(dir, tree):
    '''
    Returns all the variables available in a given .root file and with
    a specified tree
    '''
    file = uproot.open(dir)
    treename = tree_dict[tree]
    ttree = file[treename]
    ttree_data = ttree.pandas.df()

    columnlist = ttree_data.columns
    return columnlist
#a = exploreBranch(allenMVAdir+tupleTrees[1],trunks[1])
#for i in a: print(i)

