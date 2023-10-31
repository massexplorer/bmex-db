import pandas as pd
import numpy as np
from math import ceil 

Wstring = {0: '', 1: '_W1', 2: '_W2'}

def OneNSE(df, W):
    df['OneNSE'+Wstring[W]] = pd.Series(dtype=float)
    for i in range(len(df)):
        try:
            q = df.at[i, 'BE'+Wstring[W]] - float(df[(df['Z']==df.at[i,'Z']) & (df['N']==df.at[i,'N']-1)]['BE'+Wstring[W]].iloc[0])
        except:
            q = np.nan
        df.loc[i, 'OneNSE'+Wstring[W]] = q
    return df

def OnePSE(df, W):
    df['OnePSE'+Wstring[W]] = pd.Series(dtype=float)
    for i in range(len(df)):
        try:
            q = df.at[i, 'BE'+Wstring[W]] - float(df[(df['Z']==df.at[i,'Z']-1) & (df['N']==df.at[i,'N'])]['BE'+Wstring[W]].iloc[0])
        except:
            q = np.nan
        df.loc[i, 'OnePSE'+Wstring[W]] = q
    return df

def TwoNSE(df, W):
    df['TwoNSE'+Wstring[W]] = pd.Series(dtype=float)
    for i in range(len(df)):
        try:
            q = df.at[i, 'BE'+Wstring[W]] - float(df[(df['Z']==df.at[i,'Z']) & (df['N']==df.at[i,'N']-2)]['BE'+Wstring[W]].iloc[0])
        except:
            q = np.nan
        df.loc[i, 'TwoNSE'+Wstring[W]] = q
    return df

def TwoPSE(df, W):
    df['TwoPSE'+Wstring[W]] = pd.Series(dtype=float)
    for i in range(len(df)):
        try:
            q = df.at[i, 'BE'+Wstring[W]] - float(df[(df['Z']==df.at[i,'Z']-2) & (df['N']==df.at[i,'N'])]['BE'+Wstring[W]].iloc[0])
        except:
            q = np.nan
        df.loc[i, 'TwoPSE'+Wstring[W]] = q
    return df

def AlphaSE(df, W):
    df['AlphaSE'+Wstring[W]] = pd.Series(dtype=float)
    for i in range(len(df)):
        try:
            q = df.at[i, 'BE'+Wstring[W]] - float(df[(df['Z']==df.at[i,'Z']-2) & (df['N']==df.at[i,'N']-2)]['BE'+Wstring[W]].iloc[0])
        except:
            q = np.nan
        df.loc[i, 'AlphaSE'+Wstring[W]] = q
    return df

def TwoNSGap(df, W):
    df['TwoNSGap'+Wstring[W]] = pd.Series(dtype=float)
    for i in range(len(df)):
        try:
            q = df.at[i, 'TwoNSE'+Wstring[W]] - float(df[(df['Z']==df.at[i,'Z']) & (df['N']==df.at[i,'N']+2)]['TwoNSE'+Wstring[W]].iloc[0])
        except:
            q = np.nan
        df.loc[i, 'TwoNSGap'+Wstring[W]] = q
    return df

def TwoPSGap(df, W):
    df['TwoPSGap'+Wstring[W]] = pd.Series(dtype=float)
    for i in range(len(df)):
        try:
            q = df.at[i, 'TwoPSE'+Wstring[W]] - float(df[(df['Z']==df.at[i,'Z']+2) & (df['N']==df.at[i,'N'])]['TwoPSE'+Wstring[W]].iloc[0])
        except:
            q = np.nan
        df.loc[i, 'TwoPSGap'+Wstring[W]] = q
    return df

def DoubleMDiff(df, W):
    df['DoubleMDiff'+Wstring[W]] = pd.Series(dtype=float)
    for i in range(len(df)):
        try:
            q = (df.at[i, 'TwoPSE'+Wstring[W]] - float(df[(df['Z']==df.at[i,'Z']) & (df['N']==df.at[i,'N']-2)]['TwoPSE'+Wstring[W]].iloc[0]) )/4
        except:
            q = np.nan
        df.loc[i, 'DoubleMDiff'+Wstring[W]] = q
    return df

def N3PointOED(df, W):
    df['N3PointOED'+Wstring[W]] = pd.Series(dtype=float)
    for i in range(len(df)):
        try:
            q = (-1)**(df.at[i,'N'])*(df.at[i, 'OneNSE'+Wstring[W]] - float(df[(df['Z']==df.at[i,'Z']) & (df['N']==df.at[i,'N']+1)]['OneNSE'+Wstring[W]].iloc[0]) )/2
        except:
            q = np.nan
        df.loc[i, 'N3PointOED'+Wstring[W]] = q
    return df

def P3PointOED(df, W):
    df['P3PointOED'+Wstring[W]] = pd.Series(dtype=float)
    for i in range(len(df)):
        try:
            q = (-1)**(df.at[i,'Z'])*(df.at[i, 'OnePSE'+Wstring[W]] - float(df[(df['Z']==df.at[i,'Z']+1) & (df['N']==df.at[i,'N'])]['OnePSE'+Wstring[W]].iloc[0]) )/2
        except:
            q = np.nan
        df.loc[i, 'P3PointOED'+Wstring[W]] = q
    return df

def SNESplitting(df, W):
    df['SNESplitting'+Wstring[W]] = pd.Series(dtype=float)
    for i in range(len(df)):
        try:
            q = (-1)**(df.at[i,'N'])*(df.at[i, 'OneNSE'+Wstring[W]] - float(df[(df['Z']==df.at[i,'Z']) & (df['N']==df.at[i,'N']+2)]['OneNSE'+Wstring[W]].iloc[0]))
        except:
            q = np.nan
        df.loc[i, 'SNESplitting'+Wstring[W]] = q
    return df

def SPESplitting(df, W):
    df['SPESplitting'+Wstring[W]] = pd.Series(dtype=float)
    for i in range(len(df)):
        try:
            q = (-1)**(df.at[i,'Z'])*(df.at[i, 'OnePSE'+Wstring[W]] - float(df[(df['Z']==df.at[i,'Z']+2) & (df['N']==df.at[i,'N'])]['OnePSE'+Wstring[W]].iloc[0]))
        except:
            q = np.nan
        df.loc[i, 'SPESplitting'+Wstring[W]] = q
    return df

def WignerEC(df, W):
    df['WignerEC'+Wstring[W]] = pd.Series(dtype=float)
    for i in range(len(df)):
        if df.at[i,'N'] == df.at[i,'Z']:
            df.loc[i, 'WignerEC'+Wstring[W]] = np.nan
            continue
        try:
            q = (df.at[i, 'DoubleMDiff'+Wstring[W]] - (float(df[(df['Z']==df.at[i,'Z']) & \
            (df['N']==df.at[i,'N']-2)]['DoubleMDiff'+Wstring[W]].iloc[0]) - float(df[(df['Z']==df.at[i,'Z']+2) & \
            (df['N']==df.at[i,'N'])]['DoubleMDiff'+Wstring[W]].iloc[0])))
        except:
            q = np.nan
        df.loc[i, 'WignerEC'+Wstring[W]] = q
    return df    

def BEperA(df, W):
    df['BEperA'+Wstring[W]] = pd.Series(dtype=float)
    for i in range(len(df)):
        try:
            q = df.at[i, 'BE'+Wstring[W]]/(df.at[i,'Z']+df.at[i,'N'])
        except:
            q = np.nan
        df.loc[i, 'BEperA'+Wstring[W]] = q
    return df



def eOneNSE(df):
    shifted_df = df[['N','Z','eBE']].copy()
    shifted_df['N'] = shifted_df['N'].add(1)
    df = pd.merge(df, shifted_df, on=['Z', 'N'], how='left', suffixes=('', 'p'))
    df['eOneNSE'] = ((df['eBE'] + df['eBEp'])/2).apply(lambda x: ceil(x) if not np.isnan(x) else 2)
    return df

def eOnePSE(df):
    shifted_df = df[['N','Z','eBE']].copy()
    shifted_df['Z'] = shifted_df['Z'].add(1)
    df = pd.merge(df, shifted_df, on=['Z', 'N'], how='left', suffixes=('', 'p'))
    df['eOnePSE'] = ((df['eBE'] + df['eBEp'])/2).apply(lambda x: ceil(x) if not np.isnan(x) else 2)
    return df

def eTwoNSE(df):
    shifted_df = df[['N','Z','eBE']].copy()
    shifted_df['N'] = shifted_df['N'].add(2)
    df = pd.merge(df, shifted_df, on=['Z', 'N'], how='left', suffixes=('', 'p'))
    df['eTwoNSE'] = ((df['eBE'] + df['eBEp'])/2).apply(lambda x: ceil(x) if not np.isnan(x) else 2)
    return df

def eTwoPSE(df):
    shifted_df = df[['N','Z','eBE']].copy()
    shifted_df['Z'] = shifted_df['Z'].add(2)
    df = pd.merge(df, shifted_df, on=['Z', 'N'], how='left', suffixes=('', 'p'))
    df['eTwoPSE'] = ((df['eBE'] + df['eBEp'])/2).apply(lambda x: ceil(x) if not np.isnan(x) else 2)
    return df

def eAlphaSE(df):
    shifted_df = df[['N','Z','eBE']].copy()
    shifted_df['Z'] = shifted_df['Z'].add(2)
    shifted_df['N'] = shifted_df['N'].add(2)
    df = pd.merge(df, shifted_df, on=['Z', 'N'], how='left', suffixes=('', 'p'))
    df['eAlphaSE'] = ((df['eBE'] + df['eBEp'])/2).apply(lambda x: ceil(x) if not np.isnan(x) else 2)
    return df

def eTwoNSGap(df):
    try:
        df = df[['N','Z','eTwoNSE']]
    except:
        df = eTwoNSE(df)[['N','Z','eTwoNSE']]
    shifted_df = df.copy()
    shifted_df['N'] = shifted_df['N'].sub(2)
    df = pd.merge(df, shifted_df, on=['Z', 'N'], how='left', suffixes=('', 'p'))
    df['eTwoNSGap'] = ((df['eTwoNSE'] + df['eTwoNSEp'])/2).apply(lambda x: ceil(x) if not np.isnan(x) else 2)
    return df

def eTwoPSGap(df):
    try:
        df = df[['N','Z','eTwoPSE']]
    except:
        df = eTwoPSE(df)[['N','Z','eTwoPSE']]
    shifted_df = df.copy()
    shifted_df['Z'] = shifted_df['Z'].sub(2)
    df = pd.merge(df, shifted_df, on=['Z', 'N'], how='left', suffixes=('', 'p'))
    df['eTwoPSGap'] = ((df['eTwoPSE'] - df['eTwoPSEp'])/2).apply(lambda x: ceil(x) if not np.isnan(x) else 2)
    return df

def eDoubleMDiff(df):
    try:
        df = df[['N','Z','eTwoPSE']]
    except:
        df = eTwoPSE(df)[['N','Z','eTwoPSE']]
    shifted_df = df.copy()
    shifted_df['Z'] = shifted_df['Z'].add(2)
    df = pd.merge(df, shifted_df, on=['Z', 'N'], how='left', suffixes=('', 'p'))
    df['eDoubleMDiff'] = ((df['eTwoPSE'] + df['eTwoPSEp'])/2).apply(lambda x: ceil(x) if not np.isnan(x) else 2)
    return df

def eN3PointOED(df):
    try:
        df = df[['N','Z','eOneNSE']]
    except:
        df = eOneNSE(df)[['N','Z','eOneNSE']]
    shifted_df = df.copy()
    shifted_df['N'] = shifted_df['N'].sub(1)
    df = pd.merge(df, shifted_df, on=['Z', 'N'], how='left', suffixes=('', 'p'))
    df['eN3PointOED'] = ((df['eOneNSE'] + df['eOneNSEp'])/2).apply(lambda x: ceil(x) if not np.isnan(x) else 2)
    return df

def eP3PointOED(df):
    try:
        df = df[['N','Z','eOnePSE']]
    except:
        df = eOnePSE(df)[['N','Z','eOnePSE']]
    shifted_df = df.copy()
    shifted_df['Z'] = shifted_df['Z'].sub(1)
    df = pd.merge(df, shifted_df, on=['Z', 'N'], how='left', suffixes=('', 'p'))
    df['eP3PointOED'] = ((df['eOnePSE'] + df['eOnePSEp'])/2).apply(lambda x: ceil(x) if not np.isnan(x) else 2)
    return df

def eSNESplitting(df):
    try:
        df = df[['N','Z','eOneNSE']]
    except:
        df = eOneNSE(df)[['N','Z','eOneNSE']]
    shifted_df = df.copy()
    shifted_df['N'] = shifted_df['N'].sub(2)
    df = pd.merge(df, shifted_df, on=['Z', 'N'], how='left', suffixes=('', 'p'))
    df['eSNESplitting'] = ((df['eOneNSE'] + df['eOneNSEp'])/2).apply(lambda x: ceil(x) if not np.isnan(x) else 2)
    return df

def eSPESplitting(df):
    try:
        df = df[['N','Z','eOnePSE']]
    except:
        df = eOnePSE(df)[['N','Z','eOnePSE']]
    shifted_df = df.copy()
    shifted_df['Z'] = shifted_df['Z'].sub(2)
    df = pd.merge(df, shifted_df, on=['Z', 'N'], how='left', suffixes=('', 'p'))
    df['eSPESplitting'] = ((df['eOnePSE'] + df['eOnePSEp'])/2).apply(lambda x: ceil(x) if not np.isnan(x) else 2)
    return df

def eWignerEC(df):
    try:
        df = df[['N','Z','eDoubleMDiff']]
    except:
        df = eDoubleMDiff(df)[['N','Z','eDoubleMDiff']]
    shifted_df = df.copy()
    shifted_df['Z'] = shifted_df['Z'].add(2)
    shifted_df2 = df.copy()
    shifted_df2['N'] = shifted_df2['N'].sub(2)
    df = pd.merge(df, shifted_df, on=['Z', 'N'], how='left', suffixes=('', 'p'))
    df = pd.merge(df, shifted_df2, on=['Z', 'N'], how='left', suffixes=('', 'p2'))
    df['eWignerEC'] = ((df['eDoubleMDiff']+df['eDoubleMDiffp'] + df['eDoubleMDiffp2'])/3).apply(lambda x: ceil(x) if not np.isnan(x) else 2)
    df['eWignerEC'] = np.where(df['N']==df['Z'], df['eWignerEC'], np.nan)
    return df

def eBEperA(df):
    df['eBEperA'] = df['eBE']
    return df

    
def Wig1(Z, N):
    return (1.8*np.exp(-380*((N-Z)/(N+Z))**2))-(.84*abs(N-Z)*np.exp(-(((N+Z)/26)**2)))

def Wig2(Z, N):
    return -47*(abs(N-Z)/(N+Z))

def Identity(Z, N):
    return 0