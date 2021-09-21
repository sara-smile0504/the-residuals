# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:31:47 2021

@author: foster-s
"""

# This file will do the histograms for all numeric values. 
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Read in the full csv file
folder = 'C:/Users/foster-s/OneDrive - Texas A&M University/Python Projects/Humana/'
folder2 = 'C:/Users/foster-s/OneDrive - Texas A&M University/Python Projects/Humana/Visualizations'
train_df = pd.read_csv(folder+"training_clean_full.csv")

# Put full csv into a pickle file
train_df.to_pickle(folder+'training_clean_full.pickle')
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

for n in num_list:
    training_clean_full.hist(column=n)
    plt.savefig(folder+'Distribution of '+n+'.png')
    

