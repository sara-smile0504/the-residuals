# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 21:03:32 2021

@author: adesuyi-m
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import data
folder = 'C:\\Users\\adesuyi-m\\Documents\\Humana Competition\\'

training_clean_full = pd.read_csv(folder+'training_clean_full.csv')
training_clean_full.drop('Unnamed: 0',axis=1,inplace=True)
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

#make boxplots for numeric columns
for n in num_list:
    scat, ax = plt.subplots()
    ax = sns.boxplot(x='covid_vaccination',y=n,data=training_clean_full)
    ax.set_title('Boxplot of Vaccinated vs.'+n)
    scat.savefig(folder+"\\Plots\\Boxplot of Vaccinated vs."+n+".png")

#make countplots for categorical columns
for s in string_list:
    cplot, ax = plt.subplots(figsize=(30,10))
    ax = sns.countplot(x=s,hue='covid_vaccination',data=training_clean_full)
    ax.set_title('Count Plot of Vaccinated vs. '+s)
    cplot.savefig(folder+'\\Plots\\Count Plot of Vaccinated vs.'+s+'.png')
    
hm, ax = plt.subplots(figsize=(200,200))
ax = sns.heatmap(training_clean_full[num_list].corr(),annot=True,vmin=-1,center=0,vmax=1)
hm.savefig(folder+"\\Plots\\Heat Map of Numeric Variables")
