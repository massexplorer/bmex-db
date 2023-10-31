import pandas as pd
import numpy as np
from math import ceil
import os
from utils.quantities import *

def build_db(model_to_update='all'):
    version_df = pd.read_csv('db_versions.csv')
    if model_to_update!='all':
            mnames = [model_to_update]
    else:
        mnames = ['AME2020', 'ME2', 'MEdelta', 'PC1', 'NL3S', 'SKMS', 'SKP', 'SLY4', 'SV', 'UNEDF0', \
            'UNEDF1', 'UNEDF2', 'FRDM12', 'HFB24', 'BCPM', 'D1M']
        os.remove('./bmex_masses.h5')
    qfuncs = [OneNSE, OnePSE, TwoNSE, TwoPSE, AlphaSE, TwoNSGap, TwoPSGap, DoubleMDiff, N3PointOED, P3PointOED, 
            SNESplitting, SPESplitting, WignerEC, BEperA] 
    exp_qfuncs = [eOneNSE, eOnePSE, eTwoNSE, eTwoPSE, eAlphaSE, eTwoNSGap, eTwoPSGap, eDoubleMDiff, 
            eN3PointOED, eP3PointOED, eSNESplitting, eSPESplitting, eWignerEC, eBEperA]
    for mname in mnames:
        model = pd.read_csv('model_csvs/'+mname+'.csv')
        model = model.sort_values(['Z', 'N'])
        model.dropna(how='all', inplace=True)
        # ADDED POINTS
        for index in version_df.index:
            if version_df.loc[index, 'dataset'] == mname:
                Z, N, val = version_df.loc[index, 'Z'], version_df.loc[index, 'N'], version_df.loc[index, 'value']
                q = version_df.loc[index, 'q']
                if len(model.loc[(model['Z']==Z) & (model['N']==N), 'Z']) == 0: # if (Z,N) does not exist in db
                    new_row_data = {'Z': Z, 'N': N, q: val}
                    pd.concat([model, new_row_data], ignore_index=True)
                else:
                    model.loc[(model['Z']==Z) & (model['N']==N), q] = val
                try:
                    model.loc[(model['Z']==Z) & (model['N']==N), 'uBE'] = version_df.loc[index, 'uncer']
                    model.loc[(model['Z']==Z) & (model['N']==N), 'eBE'] = int(version_df.loc[index, 'estimated'])
                except:
                    pass
        qnames = ['OneNSE', 'OnePSE', 'TwoNSE', 'TwoPSE', 'AlphaSE', 'TwoNSGap', 'TwoPSGap', 'DoubleMDiff', 'N3PointOED', 
                'P3PointOED', 'SNESplitting', 'SPESplitting', 'WignerEC', 'BEperA', 'OneNSE_W1', 'OnePSE_W1', 'TwoNSE_W1', 
                'TwoPSE_W1', 'AlphaSE_W1', 'TwoNSGap_W1', 'TwoPSGap_W1', 'DoubleMDiff_W1', 'N3PointOED_W1', 'P3PointOED_W1', 
                'SNESplitting_W1', 'SPESplitting_W1', 'WignerEC_W1', 'BEperA_W1', 'OneNSE_W2', 'OnePSE_W2', 'TwoNSE_W2', 
                'TwoPSE_W2', 'AlphaSE_W2', 'TwoNSGap_W2', 'TwoPSGap_W2', 'DoubleMDiff_W2', 'N3PointOED_W2', 'P3PointOED_W2', 
                'SNESplitting_W2', 'SPESplitting_W2', 'WignerEC_W2', 'BEperA_W2']
        if mname == 'AME2020':
            qnames += ['eOneNSE', 'eOnePSE', 'eTwoNSE', 'eTwoPSE', 'eAlphaSE', 'eTwoNSGap', 'eTwoPSGap', 'eDoubleMDiff', 
                    'eN3PointOED', 'eP3PointOED', 'eSNESplitting', 'eSPESplitting', 'eWignerEC', 'eBEperA']
        model['BE_W1'] = model['BE'] - Wig1(model['Z'], model['N'])
        model['BE_W2'] = model['BE'] - Wig2(model['Z'], model['N'])
        quan_df = pd.DataFrame(columns=qnames, index=range(len(model.index)))
        for W in range(3):
            for q in qfuncs:
                quan_df[q.__name__+Wstring[W]] = q(model, W)[q.__name__+Wstring[W]]
        if mname == 'AME2020':
            for q in exp_qfuncs:
                quan_df[q.__name__] = q(model)[q.__name__]
        master_df = pd.concat([model, quan_df], axis=1)
        duplicate_columns = master_df.columns.duplicated()
        duplicate_column_names = master_df.columns[duplicate_columns] # Extract the names of the duplicate columns
        master_df = master_df.loc[:, ~duplicate_columns]
        master_df.to_hdf('./bmex_masses.h5', key=mname, index=False)
        print(mname, 'done')



