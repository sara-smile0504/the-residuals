# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:53:47 2021

@author: foster-s
"""

import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Read in the file from the pickle
folder = 'C:/Users/foster-s/OneDrive - Texas A&M University/Python Projects/Humana/'
train_subset_from_pickle = pd.read_pickle(folder+'train_subset.pickle')

# Determine which variables are actually numeric
train_subset_from_pickle.info()

# Isolate said variables, you can also use dataframe.list() function to have 
# python print the fucking thing
variables = ['auth_3mth_acute_ccs_086',
 'rx_tier_2_pmpm_ct',
 'cons_n2pwh',
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
 'rwjf_mental_distress_pct',
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
 'atlas_low_education_2015_update',
 'race_cd']

# Run a fucking for loop to print off all these mother fuckers.
for i in variables:
    his = train_subset_from_pickle.hist(column=i)
