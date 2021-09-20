# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 20:26:50 2021

@author: maddox-b
"""
import pandas as pd
import numpy as np
import pickle

original_train = pd.read_csv(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\2021_Competition_Training.csv')
t_col = original_train.columns
brandon_train  = pd.read_csv(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\data_clean\training_clean_brandon.csv')
josh_train     = pd.read_csv(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\data_clean\2021_Competition_Training_col_92_184_Josh.csv')
matthew_train  = pd.read_csv(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\data_clean\2021_Competition_Training_col_184_275_Matt.csv')
sara_train     = pd.read_csv(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\data_clean\Sara_Clean_Training_Columns.csv')

brandon_train = brandon_train.drop('Unnamed: 0', axis=1)
josh_train    = josh_train.drop('Unnamed: 0', axis=1)
inject        = pd.concat([original_train['ID'], original_train['oontwk_mbr_resp_pmpm_cost_t_6-3-0m_b4']], axis = 1)
matthew_train = matthew_train.drop(['Unnamed: 0','auth_3mth_acute_ccs_086'], axis=1)
sara_train    = sara_train.drop('Unnamed: 0', axis=1)

b_col = brandon_train.columns
j_col = josh_train.columns
m_col = matthew_train.columns
s_col = sara_train.columns

b_list = (2,9,13,16,26,29,31,51,62,75)
for each in b_list:
    i = each - 1
    brandon_train.iloc[:,i] = brandon_train.iloc[:,i].replace('*', np.nan).astype(float)
    #print(brandon_train.iloc[:,i].value_counts(dropna = False))
half          = pd.merge(brandon_train, josh_train, on='ID')
whoops        = pd.merge(half, inject, on='ID')
three_quarter = pd.merge(whoops, matthew_train, on='ID')
full          = pd.merge(three_quarter, sara_train, on='ID')

full['bh_ip_snf_net_paid_pmpm_cost_9to12m_b4'] = full['bh_ip_snf_net_paid_pmpm_cost_9to12m_b4'].replace('*', np.nan).astype(float)

full_col = full.columns
t_col = t_col.drop('Unnamed: 0')

# should print zero if the columns in the merged dataset match the columns 
# in the original dataset
q = 0
w = 0
for e in full_col:
    if e != t_col[q]:
        w += 1
    q += 1
print()
print('Non-matching columns from original dataset:', w)

# should equal zero if there are no '*' in the dataset
t = 0
for c in full.columns:
    for o in full[c]:
        if o == '*':
            t += 1
print()
print('Number of * characters in this dataset:', t)

# creates a table for the quantity missing and percent missing for each variable
total_missing    = full.isnull().sum().sort_values(ascending=False)
percent_1        = full.isnull().sum()/full.isnull().count()*100
percent_2        = (round(percent_1, 1)).sort_values(ascending=False)
#missing_data    = pd.concat([total_missing, percent_2], axis=1, keys=['Total Missing', '% Missing'])
full_dtypes      = full.dtypes
missing_w_dtypes = pd.concat([total_missing, percent_2, full_dtypes], axis=1, keys=['Total Missing', '% Missing', 'Data Type'])

zero_list = []
for col in full.columns:
    if full[col].dtype != 'O' and full[col].mean()==0 and full[col].std()==0:
        zero_list.append(col)
print()
print(zero_list)

for b in missing_w_dtypes.index:
    if b in zero_list:
        missing_w_dtypes['zero_value'].loc[b] = 0
        
missing_w_dtypes.to_csv(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\data_clean\clean training data\full_train missing.csv')

full.to_csv(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\data_clean\clean training data\training_clean_full.csv')

pickle.dump(full, open(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\data_clean\clean training data\training_clean_full.csv.pkl', 'wb'))

# i = 0
# insert = '*'*50
# for each in full_col:
#     if full[each].dtype == 'O' and i != 0:
#         print()
#         print(insert)
#         print('COLUMN:', i)
#         print(full[each].dtype)
#         print(full[each].describe())
#         print(full[each].value_counts())
#     i+=1

# i = 0
# insert = '*'*50
# for each in full_col:
#     if full[each].dtype != 'O' and full[each].mean() != 0: 
#         print()
#         print(insert)
#         print('COLUMN:', i)
#         print(full[each].dtype)
#         print(full[each].describe())
#     i+=1
























