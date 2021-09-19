# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 20:34:56 2021

@author: maddox-b
"""
## TRAINING SET COLUMNS WITH MULTIPLE DATA TYPES
# (2,8,9,11,13,16,20,26,28,29,31,33,51,55,58,62,64,66,68,75,85,102,124,127,
#  131,132,135,160,174,180,187,192,202,209,210,211,215,220,230,234,240,243,
#  247,251,255,261,285,293,297,300,305,306,309,323,334,344,345,352,353,355,359)

## TEST SET COLUMNS WITH MULTIPLE DATA TYPES
# (2,8,9,11,13,16,20,26,28,29,31,33,51,55,58,62,64,66,75,82,85,102,124,131,
#  132,135,159,173,179,191,208,209,210,219,233,239,246,254,260,284,287,292,
#  296,304,305,307,308,322,333,343,344,349,351,352,354)

trainingset = [2,8,9,11,13,16,20,26,28,29,31,33,51,55,58,62,64,66,68,75,85,102,124,127,131,132,135,160,174,180,187,192,202,209,210,211,215,220,230,234,240,243,247,251,255,261,285,293,297,300,305,306,309,323,334,344,345,352,353,355,359]
testset = [2,8,9,11,13,16,20,26,28,29,31,33,51,55,58,62,64,66,75,82,85,102,124,131,132,135,159,173,179,191,208,209,210,219,233,239,246,254,260,284,287,292,296,304,305,307,308,322,333,343,344,349,351,352,354]
for e in testset:
    trainingset.append(e)
bothset = set(trainingset)

# rows with multiple data types from either training or test set
#{2, 8, 9, 11, 13, 16, 20, 26, 28, 29, 31, 33, 51, 55, 58, 62, 64, 66, 68, 75, 
# 82, 85, 102, 124, 127, 131, 132, 135, 159, 160, 173, 174, 179, 180, 187, 191,
# 192, 202, 208, 209, 210, 211, 215, 219, 220, 230, 233, 234, 239, 240, 243, 
# 246, 247, 251, 254, 255, 260, 261, 284, 285, 287, 292, 293, 296, 297, 300, 
# 304, 305, 306, 307, 308, 309, 322, 323, 333, 334, 343, 344, 345, 349, 351, 
# 352, 353, 354, 355, 359}

import pandas as pd
import pickle
import numpy as np

insert = '*'*50
firstload = False
secondload = True
if firstload:
    train = pd.read_csv(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\2021_Competition_Training.csv')
    pickle.dump(train, open(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\2021_Competition_Training.pkl', 'wb'))
    
    test = pd.read_csv(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\2021_Competition_Holdout.csv')
    pickle.dump(test, open(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\2021_Competition_Holdout.pkl', 'wb'))

    b_train = train.drop('Unnamed: 0', axis=1)
    b_train = b_train.drop(b_train.columns[91:367], axis=1)
    pickle.dump(b_train, open(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\b_train.pkl', 'wb'))

    b_test = test.drop('Unnamed: 0', axis=1)
    b_test = b_test.drop(b_test.columns[91:366], axis=1)
    pickle.dump(b_test, open(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\b_test.pkl', 'wb'))

if secondload:
    b_train = pickle.load(open(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\b_train.pkl', 'rb'))
    b_test  = pickle.load(open(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\b_test.pkl', 'rb'))

train_columns = b_train.columns
test_columns = b_test.columns
    
train_dtypes = b_train.dtypes
test_dtypes = b_test.dtypes

i = 0
for each in train_columns:
    print()
    print(insert)
    print('COLUMN:', i)
    print(b_train[each].dtype)
    print(b_train[each].describe())
    if b_train[each].dtype != 'float64' and i != 0:
        print()
        print(b_train[each].value_counts())
    i+=1

train_test = [b_train, b_test]

# 10
for data in train_test:
    data['src_div_id'] = data['src_div_id'].replace('*', np.NaN)
    data['src_div_id'] = data['src_div_id'].astype('float64')
    
# 14    
for data in train_test:
    data['mcc_ano_pmpm_ct_t_9-6-3m_b4'] = data['mcc_ano_pmpm_ct_t_9-6-3m_b4'].replace('*', np.NaN)
    
# 19
for data in train_test:
    data['rx_gpi4_6110_pmpm_ct'] = data['rx_gpi4_6110_pmpm_ct'].replace('*', np.NaN)
    data['rx_gpi4_6110_pmpm_ct'] = data['rx_gpi4_6110_pmpm_ct'].astype('float64')

# 27
for data in train_test:
    data['rx_bh_pmpm_ct_0to3m_b4'] = data['rx_bh_pmpm_ct_0to3m_b4'].replace('*', np.NaN)
    data['rx_bh_pmpm_ct_0to3m_b4'] = data['rx_bh_pmpm_ct_0to3m_b4'].astype('float64') 

# 32    
for data in train_test:
    data['auth_3mth_dc_home'] = data['auth_3mth_dc_home'].replace('*', np.NaN)
    data['auth_3mth_dc_home'] = data['auth_3mth_dc_home'].astype('float64') 

# 34
for data in train_test:
    data['rx_gpi2_17_pmpm_cost_t_12-9-6m_b4'] = data['rx_gpi2_17_pmpm_cost_t_12-9-6m_b4'].replace('*', np.NaN)    

# 54
for data in train_test:
    data['auth_3mth_dc_no_ref'] = data['auth_3mth_dc_no_ref'].replace('*', np.NaN)
    data['auth_3mth_dc_no_ref'] = data['auth_3mth_dc_no_ref'].astype('float64') 
    
# 57
for data in train_test:
    data['auth_3mth_dc_snf'] = data['auth_3mth_dc_snf'].replace('*', np.NaN)
    data['auth_3mth_dc_snf'] = data['auth_3mth_dc_snf'].astype('float64')     

# 63
for data in train_test:
    data['auth_3mth_psychic'] = data['auth_3mth_psychic'].replace('*', np.NaN)
    data['auth_3mth_psychic'] = data['auth_3mth_psychic'].astype('float64')

# 65
for data in train_test:
    data['auth_3mth_bh_acute'] = data['auth_3mth_bh_acute'].replace('*', np.NaN)
    data['auth_3mth_bh_acute'] = data['auth_3mth_bh_acute'].astype('float64')
    
# 67
for data in train_test:
    data['auth_3mth_acute_chf'] = data['auth_3mth_acute_chf'].replace('*', np.NaN)
    data['auth_3mth_acute_chf'] = data['auth_3mth_acute_chf'].astype('float64')

# 72
for data in train_test:
    data['bh_urgent_care_copay_pmpm_cost_t_12-9-6m_b4'] = data['bh_urgent_care_copay_pmpm_cost_t_12-9-6m_b4'].replace('*', np.NaN)    

# 82 - issue in test set only
for data in train_test:
    data['auth_3mth_acute_skn'] = data['auth_3mth_acute_skn'].replace('*', np.NaN)
    data['auth_3mth_acute_skn'] = data['auth_3mth_acute_skn'].astype('float64')
    
# 84
for data in train_test:
    data['rx_gpi2_34_dist_gpi6_pmpm_ct'] = data['rx_gpi2_34_dist_gpi6_pmpm_ct'].replace('*', np.NaN)
    data['rx_gpi2_34_dist_gpi6_pmpm_ct'] = data['rx_gpi2_34_dist_gpi6_pmpm_ct'].astype('float64')

# calculate the total missing from each column and display as a table from 
# greatest to least
total_missing = b_train.isnull().sum().sort_values(ascending=False)
percent_1 = b_train.isnull().sum()/b_train.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total_missing, percent_2], axis=1, keys=['Total Missing', '% Missing'])

missing_data.to_csv(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\b_train missing.csv')

# reviewing each object variable to note the possible values for imputation
i = 0
for each in train_columns:
    if b_train[each].dtype == 'O' and i != 0:
        print()
        print(insert)
        print('COLUMN:', i)
        print(b_train[each].unique())
    i+=1
    
b_train.to_csv(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\training_clean_brandon.csv')
b_test.to_csv(r'C:\Users\maddox-b\OneDrive - Texas A&M University\Documents\Humana 2021\test_clean_brandon.csv')














