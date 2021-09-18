# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 18:12:00 2021

@author: adesuyi-m
"""

import numpy as np
import pandas as pd

folder = 'C:\\Users\\adesuyi-m\\Documents\\Humana Competition\\'
train_file = '2021_Competition_Training.csv'
test_file = '2021_Competition_Holdout.csv'

train = pd.read_csv(folder+train_file)
test = pd.read_csv(folder+test_file)

train_pkl = train.to_pickle(folder+'2021_Competition_Training.pkl')
train = pd.read_pickle(folder+'2021_Competition_Training.pkl')

dirty_columns = [2,8,9,11,13,16,20,26,28,29,31,33,51,55,58,62,64,66,68,75,85,102,124,127,131,132,135,160,174,180,187,192,202,209,210,211,215,220,230,234,240,243,247,251,255,261,285,293,297,300,305,306,309,323,334,344,345,352,353,355,359]


brandon_list = list(range(1,92))
josh_list = list(range(92,184))
matt_list = list(range(184,276))
sara_list = list(range(276,368))

group_list = [brandon_list,josh_list,matt_list,sara_list]

for l in group_list:
    l.insert(0,1)


train.iloc[:,184:276].to_pickle(folder+'2021_Competition_Training_col_184_275_Matt.pkl')
brandon_clean = train.iloc[:,brandon_list]
josh_clean = train.iloc[:,josh_list]
matt_clean = train.iloc[:,matt_list]
sara_clean = train.iloc[:,sara_list]

brandon_dirty_columns = ['auth_3mth_post_acute_dia', 'bh_ip_snf_net_paid_pmpm_cost_9to12m_b4',
       'auth_3mth_acute_ckd', 'src_div_id',
       'bh_ip_snf_net_paid_pmpm_cost_3to6m_b4', 'auth_3mth_post_acute_trm',
       'rx_gpi4_6110_pmpm_ct', 'auth_3mth_acute_vco', 'rx_bh_pmpm_ct_0to3m_b4',
       'auth_3mth_dc_ltac', 'auth_3mth_post_acute_inj', 'auth_3mth_dc_home',
       'bh_ip_snf_mbr_resp_pmpm_cost_6to9m_b4', 'auth_3mth_dc_no_ref',
       'auth_3mth_dc_snf', 'bh_ip_snf_net_paid_pmpm_cost_0to3m_b4',
       'auth_3mth_psychic', 'auth_3mth_bh_acute', 'auth_3mth_acute_chf',
       'auth_3mth_acute_bld', 'rx_gpi2_34_dist_gpi6_pmpm_ct']

josh_dirty_columns = ['lab_albumin_loinc_pmpm_ct', 'rx_gpi2_72_pmpm_ct_6to9m_b4',
       'auth_3mth_acute_res', 'auth_3mth_acute_dig',
       'auth_3mth_dc_acute_rehab', 'bh_ip_snf_mbr_resp_pmpm_cost_3to6m_b4',
       'auth_3mth_non_er', 'bh_ip_snf_mbr_resp_pmpm_cost_9to12m_b4',
       'auth_3mth_post_acute_cer']

matt_dirty_columns = ['auth_3mth_post_acute_mus', 'bh_ip_snf_net_paid_pmpm_cost_6to9m_b4',
       'auth_3mth_post_acute_sns', 'auth_3mth_acute_can',
       'auth_3mth_post_acute', 'auth_3mth_facility',
       'auth_3mth_post_acute_men', 'auth_3mth_home', 'auth_3mth_transplant',
       'rev_cms_ansth_pmpm_ct', 'auth_3mth_acute', 'auth_3mth_dc_left_ama',
       'auth_3mth_acute_ccs_227', 'auth_3mth_dc_custodial',
       'rx_gpi2_90_dist_gpi6_pmpm_ct_9to12m_b4', 'auth_3mth_ltac']

sara_dirty_columns = ['auth_3mth_snf_post_hsp', 'auth_3mth_acute_trm',
       'rej_total_physician_office_visit_ct_pmpm_0to3m_b4',
       'auth_3mth_snf_direct', 'auth_3mth_dc_home_health',
       'rx_gpi2_56_dist_gpi6_pmpm_ct_3to6m_b4', 'auth_3mth_acute_ner',
       'ccsp_065_pmpm_ct', 'auth_3mth_post_er', 'rx_gpi2_33_pmpm_ct_0to3m_b4',
       'auth_3mth_post_acute_chf', 'auth_3mth_dc_other',
       'auth_3mth_bh_acute_mean_los', 'auth_3mth_post_acute_gus',
       'auth_3mth_acute_mus']

matt_clean_cols = matt_clean.dtypes.reset_index




#Used to see unique values in the dirty columns
for col in dirty_columns:
    print(matt_clean[col].unique())


#Code to clean Matt's columns
matt_clean = matt_clean.replace('*',np.nan)

#Convert to float
matt_clean[matt_dirty_columns] = matt_clean[matt_dirty_columns].astype(float)


#Export to pickle file
matt_clean.to_pickle(folder+'2021_Competition_Training_col_184_275_Matt.pkl')
matt_clean = pd.read_pickle(folder+'2021_Competition_Training_col_184_275_Matt.pkl')
matt_clean.to_csv(folder+'2021_Competition_Training_col_184_275_Matt.csv')

#Gives mean, median, mode, range
describe = matt_clean.describe()


#Gives Data Types for each column
dtypes = matt_clean.dtypes
dtypes.to_excel(folder+"Matt Columns Data Types.xlsx")
