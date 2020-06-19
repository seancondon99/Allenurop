import sys, os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import uproot

tree_dict = {
    'N1Trk' : 'DecayTreeTuple/N1Trk',
    'N2Trk' : 'DecayTreeTuple#1/N2Trk',
    'N3Trk' : 'DecayTreeTuple#2/N3Trk',
    'N4Trk' : 'DecayTreeTuple#3/N4Trk'
}

def parse_args():
    '''
    Parse the command line arguments.
    '''
    parser = argparse.ArgumentParser(
        description=''' 
Script for plotting variables from TTrees. Uses uproot to create a
Pandas DataFrame from the TTree and uses matplotlib to plot the
specified variable.  
        ''')
    parser.add_argument('--input', type=str, default=None, required=True,
                        help='Input ROOT file.')
    parser.add_argument('--tree', type=str, default='N1Trk', required=True,
                        help='Input TTree name. Can be N1Trk, N2Trk, N3Trk, or N4Trk.')
    parser.add_argument('--var', type=str, default=None, required=False,
                        help='Branch to plot. If blank, print a list of branches.')
    return parser.parse_args()

def make_hist(df, var, ax, **kwargs):
    '''
    Plot a histogram.
    '''
    head = var.split('_')[0]
    sig = df[df[head+'_signal_type']>0][var]
    bkg = df[df[head+'_signal_type']==0][var]
    return ax.hist((sig, bkg), color=['tab:red','tab:blue'],
                   label=['signal','background'],
                   **kwargs)

if __name__=='__main__':
    # Read the data.
    args = parse_args()
    f = uproot.open(args.input)
    # Get the name of the TTree in the ROOT file.
    tree_name = tree_dict[args.tree]
    # Get the TTree.
    t = f[tree_name]
    # Create a pandas DataFrame from the TTree.
    df = t.pandas.df()
    
    # Print the branches of the TTree if no variable is provided.
    if args.var==None:
        print(df.columns)
    # Plot the variable.
    else:
        var = args.var
        plt.clf()
        ax = plt.gca()
        
        # This provides a sensible range for most branches, but not
        # for e.g. IP chi2 and FD chi2.
        limits = (df.quantile(0.01)[var], df.quantile(0.99)[var])

        make_hist(df, var, ax, density=True, histtype='step', range=limits, bins=50)
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_xlabel(var)
        ax.legend()
        fname = args.input.split('/')[-1]
        fname = fname.replace('.root','')
        plt.savefig(fname + '_' + var + '.pdf')
