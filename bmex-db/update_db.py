#!/usr/bin/env python

import argparse
from sys import exit
import pandas as pd
import numpy as np
import datetime
from utils.build_db import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Updates version of db and can add points to exp data')
    parser.add_argument('descr', type=str, help='Description of update') 
    parser.add_argument('--major', '-m', dest='major', default=False, action='store_true', help='Is a major version update (big change in csv\'s)')
    parser.add_argument('--value', '-v', default=None, type=float, help='New value in default units (usually MeV)')
    parser.add_argument('--uncertainty', '-u', default=None, type=float, help='New uncertainty in default units (usually MeV)')
    parser.add_argument('--estimated', '-e', dest='estimated', default=False, action='store_true', help='New value is estimated')
    parser.add_argument('--protons', '-z', default=None, type=float, help='Proton number of new value')
    parser.add_argument('--neutrons', '-n', default=None, type=float, help='Neutron number of new value')
    parser.add_argument('--dataset', '-d', default='AME2020', type=str, help='The new modified dataset')
    parser.add_argument('--quantity', '-q', default='BE', type=str, help='The new quantity')
    parser.add_argument('--ref', '-r', default=None, type=str, help='Reference to the new value\'s source')
    args = parser.parse_args()

    date = datetime.date.today().strftime('%m-%d-%Y')

    # Exception handling
    if args.dataset not in ['AME2020', 'ME2', 'MEdelta', 'PC1', 'NL3S', 'SKMS', 'SKP', 'SLY4', 'SV', 'UNEDF0', 'UNEDF1', 'UNEDF2', 'FRDM12', 'HFB24', 'BCPM', 'D1M']:
        raise ValueError("Please enter a supported dataset")

    # Read in versions csv or create it
    try: 
        df = pd.read_csv('db_versions.csv')
    except:
        df = pd.DataFrame(columns=['major', 'minor', 'date', 'value', 'uncer', 'estimated', 'Z', 'N', 'dataset', 'q', 'ref', 'descr'])
        df.loc[len(df.index)] = [1, 1, date, None, None, None, None, None, None, None, None, 'First Version']
        df.to_csv('db_versions.csv', index=None)
        exit()

    # Generate new version info
    if args.major:
        new_row = [df.loc[len(df.index)-1, 'major']+1, 1, date, args.value, args.uncertainty, args.estimated, args.protons, args.neutrons, args.dataset, args.quantity, args.ref, args.descr]
    else:
        new_row = [df.loc[len(df.index)-1, 'major'], df.loc[len(df.index)-1, 'minor']+1, date, args.value, args.uncertainty, args.estimated, args.protons, args.neutrons, args.dataset, args.quantity, args.ref, args.descr]

    # Write new versions csv
    df.loc[len(df.index)] = new_row
    df.to_csv('db_versions.csv', index=None)

    # Build new db
    if args.major:
        build_db()
    else:
        build_db(args.dataset)

