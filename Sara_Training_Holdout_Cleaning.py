# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:19:04 2021

@author: foster-s
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.Regression import linreg,stepwise

# Reading in the training data set from the file

folder = 'C:/Users/foster-s/OneDrive - Texas A&M University/Python Projects/Humana/'

train_df = pd.read_csv(folder+"2021_Training_Data.csv")
test_df = pd.read_csv(folder+"2021_Competition_Holdout.csv")

# There are several data type issues within the columns. Specifically for the 
# following columns: (2,8,9,11,13,16,20,26,28,29,31,33,51,55,58,62,64,66,68,75,
# 85,102,124,127,131,132,135,160,174,180,187,192,202,209,210,211,215,220,230,
# 234,240,243,247,251,255,261,285,293,297,300,305,306,309,323,334,344,345,352,
# 53,355,359) have mixed types.Specify dtype option on import or 
# set low_memory=False.

# Listing the columns in the training and test data file to make sure they're 
# the same
list(train_df.columns)
list(test_df.columns)

# Will be cleaning the last 92 columns in the file. Starting with auth_3mth_acute
# _ccs_086 to race_cd

# Create a subset of the training data columns along with ID so we keep 
# everything tied together
train_subset = train_df[['ID','auth_3mth_acute_ccs_086',
 'rx_tier_2_pmpm_ct',
 'cons_n2pwh',
 'rx_nonmaint_dist_gpi6_pmpm_ct_t_12-9-6m_b4',
 'atlas_berry_acrespth12',
 'atlas_pct_fmrkt_credit16',
 'atlas_slhouse12',
 'atlas_pc_fsrsales12',
 'credit_hh_1stmtgcredit',
 'auth_3mth_snf_post_hsp',
 'atlas_pct_fmrkt_wiccash16',
 'atlas_foodinsec_13_15',
 'auth_3mth_acute_cer',
 'cons_rxadhm',
 'atlas_fmrktpth16',
 'rx_nonotc_pmpm_cost_t_6-3-0m_b4',
 'cci_dia_m_pmpm_ct',
 'auth_3mth_acute_trm',
 'cons_n2phi',
 'bh_physician_office_copay_pmpm_cost_6to9m_b4',
 'rwjf_income_inequ_ratio',
 'rej_total_physician_office_visit_ct_pmpm_0to3m_b4',
 'auth_3mth_acute_dia',
 'credit_num_nonmtgcredit_60dpd',
 'auth_3mth_snf_direct',
 'credit_bal_autofinance_new',
 'auth_3mth_acute_ccs_067',
 'auth_3mth_acute_ccs_043',
 'rwjf_men_hlth_prov_ratio',
 'auth_3mth_dc_home_health',
 'rx_gpi2_56_dist_gpi6_pmpm_ct_3to6m_b4',
 'cmsd2_sns_genitourinary_pmpm_ct',
 'auth_3mth_acute_cir',
 'auth_3mth_acute_ner',
 'auth_3mth_acute_ccs_094',
 'med_ambulance_coins_pmpm_cost_t_9-6-3m_b4',
 'hedis_dia_hba1c_ge9',
 'bh_ncal_pct',
 'atlas_pct_snap16',
 'ccsp_227_pct',
 'atlas_ghveg_sqftpth12',
 'rx_days_since_last_script_6to9m_b4',
 'atlas_orchard_acrespth12',
 'atlas_persistentchildpoverty_1980_2011',
 'auth_3mth_post_acute_cad',
 'atlas_pct_laccess_multir15',
 'cons_cgqs',
 'ccsp_065_pmpm_ct',
 'auth_3mth_acute_ccs_044',
 'atlas_medhhinc',
 'rx_maint_net_paid_pmpm_cost_t_9-6-3m_b4',
 'rwjf_mental_distress_pct',
 'bh_ip_snf_admit_days_pmpm_t_9-6-3m_b4',
 'rx_phar_cat_cvs_pmpm_ct_t_9-6-3m_b4',
 'zip_cd',
 'auth_3mth_post_acute_ckd',
 'atlas_pct_laccess_nhpi15',
 'auth_3mth_post_acute_ner',
 'auth_3mth_post_er',
 'credit_num_consumerfinance_new',
 'rx_gpi2_49_pmpm_cost_0to3m_b4',
 'cons_chva',
 'atlas_avghhsize',
 'rx_overall_net_paid_pmpm_cost_6to9m_b4',
 'atlas_ownhomepct',
 'atlas_orchard_farms12',
 'total_physician_office_visit_ct_pmpm_t_6-3-0m_b4',
 'atlas_pct_fmrkt_wic16',
 'rx_gpi2_33_pmpm_ct_0to3m_b4',
 'auth_3mth_post_acute_chf',
 'rwjf_social_associate_rate',
 'atlas_freshveg_farms12',
 'auth_3mth_acute_ccs_042',
 'auth_3mth_post_acute_inf',
 'auth_3mth_acute_sns',
 'days_since_last_clm_0to3m_b4',
 'auth_3mth_dc_other',
 'auth_3mth_bh_acute_mean_los',
 'mcc_end_pct',
 'auth_3mth_post_acute_gus',
 'cons_lwcm07',
 'atlas_pct_fmrkt_otherfood16',
 'auth_3mth_post_acute_end',
 'auth_3mth_acute_mus',
 'atlas_perpov_1980_0711',
 'atlas_pct_laccess_white15',
 'auth_3mth_post_acute_mean_los',
 'rx_gpi2_66_pmpm_ct',
 'auth_3mth_acute_gus',
 'rx_generic_dist_gpi6_pmpm_ct_t_9-6-3m_b4',
 'atlas_low_education_2015_update',
 'race_cd']]

# Put the training data columns in a pickle file. 
train_subset.to_pickle(folder+'train_subset.pickle')

# Create a subset of the necessary columns from the holdout data set. 
test_subset = test_df[['ID','auth_3mth_acute_ccs_086',
 'rx_tier_2_pmpm_ct',
 'cons_n2pwh',
 'rx_nonmaint_dist_gpi6_pmpm_ct_t_12-9-6m_b4',
 'atlas_berry_acrespth12',
 'atlas_pct_fmrkt_credit16',
 'atlas_slhouse12',
 'atlas_pc_fsrsales12',
 'credit_hh_1stmtgcredit',
 'auth_3mth_snf_post_hsp',
 'atlas_pct_fmrkt_wiccash16',
 'atlas_foodinsec_13_15',
 'auth_3mth_acute_cer',
 'cons_rxadhm',
 'atlas_fmrktpth16',
 'rx_nonotc_pmpm_cost_t_6-3-0m_b4',
 'cci_dia_m_pmpm_ct',
 'auth_3mth_acute_trm',
 'cons_n2phi',
 'bh_physician_office_copay_pmpm_cost_6to9m_b4',
 'rwjf_income_inequ_ratio',
 'rej_total_physician_office_visit_ct_pmpm_0to3m_b4',
 'auth_3mth_acute_dia',
 'credit_num_nonmtgcredit_60dpd',
 'auth_3mth_snf_direct',
 'credit_bal_autofinance_new',
 'auth_3mth_acute_ccs_067',
 'auth_3mth_acute_ccs_043',
 'rwjf_men_hlth_prov_ratio',
 'auth_3mth_dc_home_health',
 'rx_gpi2_56_dist_gpi6_pmpm_ct_3to6m_b4',
 'cmsd2_sns_genitourinary_pmpm_ct',
 'auth_3mth_acute_cir',
 'auth_3mth_acute_ner',
 'auth_3mth_acute_ccs_094',
 'med_ambulance_coins_pmpm_cost_t_9-6-3m_b4',
 'hedis_dia_hba1c_ge9',
 'bh_ncal_pct',
 'atlas_pct_snap16',
 'ccsp_227_pct',
 'atlas_ghveg_sqftpth12',
 'rx_days_since_last_script_6to9m_b4',
 'atlas_orchard_acrespth12',
 'atlas_persistentchildpoverty_1980_2011',
 'auth_3mth_post_acute_cad',
 'atlas_pct_laccess_multir15',
 'cons_cgqs',
 'ccsp_065_pmpm_ct',
 'auth_3mth_acute_ccs_044',
 'atlas_medhhinc',
 'rx_maint_net_paid_pmpm_cost_t_9-6-3m_b4',
 'rwjf_mental_distress_pct',
 'bh_ip_snf_admit_days_pmpm_t_9-6-3m_b4',
 'rx_phar_cat_cvs_pmpm_ct_t_9-6-3m_b4',
 'zip_cd',
 'auth_3mth_post_acute_ckd',
 'atlas_pct_laccess_nhpi15',
 'auth_3mth_post_acute_ner',
 'auth_3mth_post_er',
 'credit_num_consumerfinance_new',
 'rx_gpi2_49_pmpm_cost_0to3m_b4',
 'cons_chva',
 'atlas_avghhsize',
 'rx_overall_net_paid_pmpm_cost_6to9m_b4',
 'atlas_ownhomepct',
 'atlas_orchard_farms12',
 'total_physician_office_visit_ct_pmpm_t_6-3-0m_b4',
 'atlas_pct_fmrkt_wic16',
 'rx_gpi2_33_pmpm_ct_0to3m_b4',
 'auth_3mth_post_acute_chf',
 'rwjf_social_associate_rate',
 'atlas_freshveg_farms12',
 'auth_3mth_acute_ccs_042',
 'auth_3mth_post_acute_inf',
 'auth_3mth_acute_sns',
 'days_since_last_clm_0to3m_b4',
 'auth_3mth_dc_other',
 'auth_3mth_bh_acute_mean_los',
 'mcc_end_pct',
 'auth_3mth_post_acute_gus',
 'cons_lwcm07',
 'atlas_pct_fmrkt_otherfood16',
 'auth_3mth_post_acute_end',
 'auth_3mth_acute_mus',
 'atlas_perpov_1980_0711',
 'atlas_pct_laccess_white15',
 'auth_3mth_post_acute_mean_los',
 'rx_gpi2_66_pmpm_ct',
 'auth_3mth_acute_gus',
 'rx_generic_dist_gpi6_pmpm_ct_t_9-6-3m_b4',
 'atlas_low_education_2015_update',
 'race_cd']]

# Push the specified columns into their own pickle file. 
test_subset.to_pickle(folder+'test_subset.pickle')

# Read both pickle files so as to use them in cleaning. 
train_subset_from_pickle = pd.read_pickle(folder+'train_subset.pickle')
test_subset_from_pickle = pd.read_pickle(folder+'test_subset.pickle')

# Let's examine the information for each file. 
print(train_subset_from_pickle.info())
print(test_subset_from_pickle.info())

# Let's take a look at all the columns in the training set
insert = '*'*50
i = 0
for column in train_subset_from_pickle:
    print()
    print(insert)
    print('COLUMN:', i)
    print(train_subset_from_pickle[column].dtype)
    print(train_subset_from_pickle[column].describe())
    if train_subset_from_pickle[column].dtype != 'float64' and i != 0:
        print()
        print(train_subset_from_pickle[column].value_counts(dropna=False))
    i+=1
    
# Some columns have * so that will need to be replaced. As seen below.
train_clean = train_subset_from_pickle.replace('*',np.nan)

# Create a list of the columns that python said did not have matching data 
# types and conver them all to float. 
dirty_dtype_columns = ['auth_3mth_snf_post_hsp', 'auth_3mth_acute_trm',
       'rej_total_physician_office_visit_ct_pmpm_0to3m_b4',
       'auth_3mth_snf_direct', 'auth_3mth_dc_home_health',
       'rx_gpi2_56_dist_gpi6_pmpm_ct_3to6m_b4', 'auth_3mth_acute_ner',
       'ccsp_065_pmpm_ct', 'auth_3mth_post_er', 'rx_gpi2_33_pmpm_ct_0to3m_b4',
       'auth_3mth_post_acute_chf', 'auth_3mth_dc_other',
       'auth_3mth_bh_acute_mean_los', 'auth_3mth_post_acute_gus',
       'auth_3mth_acute_mus']

train_clean[dirty_dtype_columns] = train_clean[dirty_dtype_columns].astype(float)

# Pushing the clean data into another pickle file then converting it into a csv
train_clean.to_pickle(folder+'Sara_Clean_Columns.pkl')
train_clean = pd.read_pickle(folder+'Sara_Clean_Columns.pkl')
train_clean.to_csv(folder+'Sara_Clean_Columns.csv')

# Let's take a look at all the columns in the test set
insert = '*'*50
i = 0
for column in test_subset_from_pickle:
    print()
    print(insert)
    print('COLUMN:', i)
    print(test_subset_from_pickle[column].dtype)
    print(test_subset_from_pickle[column].describe())
    if test_subset_from_pickle[column].dtype != 'float64' and i != 0:
        print()
        print(test_subset_from_pickle[column].value_counts(dropna=False))
    i+=1

# Same problem as the holdout there are columns with * as values, need to
# replace these
test_clean = test_subset_from_pickle.replace('*',np.nan)

# Will reuse the same columns that we know have inconsistent data types and 
# convert them to float. 
test_clean[dirty_dtype_columns] = test_clean[dirty_dtype_columns].astype(float)

# Push the clean test data into a pickle file then to a csv
test_clean.to_pickle(folder+'Sara_Clean_Holdout_Columns.pkl')
test_clean = pd.read_pickle(folder+'Sara_Clean_Holdout_Columns.pkl')
test_clean.to_csv(folder+'Sara_Clean_Holdout_Columns.csv')
