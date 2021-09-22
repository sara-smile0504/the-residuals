# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 16:03:06 2021

@author: foster-s
"""
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

folder = 'C:/Users/foster-s/OneDrive - Texas A&M University/Python Projects/Humana/'

training_clean_full = pd.read_pickle(folder+'training_clean_full.pickle')

# Dropping the unnamed column
training_clean_full.drop('Unnamed: 0',axis=1,inplace=True)
# Replacing any string data sets with nan for *
training_clean_full = training_clean_full.replace('*',np.nan)

#Gets rid of columns with only zeroes
zero_list = []
desc = training_clean_full.describe()
for c in desc.columns:
    if desc[c][1] == 0.0:
        if desc[c][2] == 0.0:
            zero_list.append(c)

training_clean_full.drop(zero_list,axis=1,inplace=True)

#make a list of numeric and string columns
string_list = [l for l in dict(training_clean_full.dtypes) if training_clean_full.dtypes[l] == 'object'][1:]
num_list = [l for l in dict(training_clean_full.dtypes) if training_clean_full.dtypes[l] in ['int64','float64']]
string_list.remove('covid_vaccination')

# Running the square root transformation
sq_root = np.sqrt(training_clean_full[num_list])

for n in num_list:
    sq_root.hist(column = n)
    plt.savefig(folder+'Sqrt Distribution of '+n+'.png')

# log transformation of the distribution
log = np.log(training_clean_full[num_list])

for n in num_list:
    log.hist(column = n)
    plt.savefig(folder+'Sqrt Distribution of '+n+'.png')

