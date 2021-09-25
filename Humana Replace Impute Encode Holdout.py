# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 21:34:28 2021

@author: adesuyi-m
"""

#AdvancedAnalytics Packages
from AdvancedAnalytics.Forest import forest_classifier
from AdvancedAnalytics.Tree import tree_classifier
from AdvancedAnalytics.Regression import logreg, stepwise
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.NeuralNetwork import nn_classifier

#SKLearn Packages
from sklearn.metrics         import f1_score,precision_score,recall_score,confusion_matrix,plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

#Import Python Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

folder = 'C:\\Users\\adesuyi-m\\Documents\\Humana Competition\\'
holdout_clean_full = pd.read_csv(folder+'holdout_clean_full.csv')
holdout_clean_full.drop('Unnamed: 0',axis=1,inplace=True)

zero_list = []
desc = holdout_clean_full.describe()
for c in desc.columns:
    if desc[c][1] == 0.0:
        if desc[c][2] == 0.0:
            zero_list.append(c)

holdout_clean_full.drop(zero_list,axis=1,inplace=True)

holdout_clean_full = holdout_clean_full.replace('*',np.nan)



string_list = [l for l in dict(holdout_clean_full.dtypes) if holdout_clean_full.dtypes[l] == 'object'][1:]
num_list = [l for l in dict(holdout_clean_full.dtypes) if holdout_clean_full.dtypes[l] in ['int64','float64']]


bn_list = ['bh_ncdm_ind','atlas_retirement_destination_2015_upda','atlas_hiamenity',
           'atlas_hipov_1115','atlas_type_2015_mining_no',
           'atlas_low_employment_2015_update','bh_ncal_ind','atlas_type_2015_recreation_no',
           'atlas_population_loss_2015_update','atlas_farm_to_school13','sex_cd',
           'atlas_persistentchildpoverty_1980_2011','atlas_perpov_1980_0711'
           ,'atlas_low_education_2015_update']

nom_list = ['src_div_id','total_bh_copay_pmpm_cost_t_9-6-3m_b4',
               'mcc_ano_pmpm_ct_t_9-6-3m_b4','rx_maint_pmpm_cost_t_12-9-6m_b4',
               'rx_nonbh_pmpm_cost_t_9-6-3m_b4','rx_gpi2_17_pmpm_cost_t_12-9-6m_b4','rx_generic_pmpm_cost_t_6-3-0m_b4',
               'rx_overall_mbr_resp_pmpm_cost_t_6-3-0m_b4','rx_overall_dist_gpi6_pmpm_ct_t_6-3-0m_b4',
               'rx_phar_cat_humana_pmpm_ct_t_9-6-3m_b4',
               'rx_overall_gpi_pmpm_ct_t_6-3-0m_b4','mcc_chf_pmpm_ct_t_9-6-3m_b4',
               'bh_urgent_care_copay_pmpm_cost_t_12-9-6m_b4','rx_maint_pmpm_cost_t_6-3-0m_b4',
               'cons_mobplus','rx_maint_net_paid_pmpm_cost_t_12-9-6m_b4',
               'rej_med_outpatient_visit_ct_pmpm_t_6-3-0m_b4',
               'rej_med_ip_snf_coins_pmpm_cost_t_9-6-3m_b4','med_physician_office_allowed_pmpm_cost_t_9-6-3m_b4',
               'total_physician_office_net_paid_pmpm_cost_t_9-6-3m_b4',
               'rx_branded_pmpm_ct_t_6-3-0m_b4','med_outpatient_deduct_pmpm_cost_t_9-6-3m_b4',
               'total_allowed_pmpm_cost_t_9-6-3m_b4',
               'cms_orig_reas_entitle_cd','oontwk_mbr_resp_pmpm_cost_t_6-3-0m_b4',
               'hum_region','rx_nonmail_dist_gpi6_pmpm_ct_t_9-6-3m_b4',
               'rej_med_er_net_paid_pmpm_cost_t_9-6-3m_b4','med_outpatient_mbr_resp_pmpm_cost_t_9-6-3m_b4',
               'rx_nonbh_net_paid_pmpm_cost_t_6-3-0m_b4',
               'rx_gpi2_39_pmpm_cost_t_6-3-0m_b4','atlas_type_2015_update',
               'total_ip_maternity_net_paid_pmpm_cost_t_12-9-6m_b4',
               'rx_maint_pmpm_ct_t_6-3-0m_b4','rx_mail_net_paid_pmpm_cost_t_6-3-0m_b4',
               'total_physician_office_mbr_resp_pmpm_cost_t_9-6-3m_b4',
               'rx_mail_mbr_resp_pmpm_cost_t_9-6-3m_b4',
               'med_outpatient_visit_ct_pmpm_t_12-9-6m_b4','rx_nonbh_pmpm_ct_t_9-6-3m_b4',
               'total_med_net_paid_pmpm_cost_t_6-3-0m_b4','rx_gpi2_62_pmpm_cost_t_9-6-3m_b4',
               'rx_overall_gpi_pmpm_ct_t_12-9-6m_b4','cons_hhcomp',
               'rx_nonmaint_dist_gpi6_pmpm_ct_t_12-9-6m_b4','rx_nonotc_pmpm_cost_t_6-3-0m_b4',
               'med_ambulance_coins_pmpm_cost_t_9-6-3m_b4',
               'rx_maint_net_paid_pmpm_cost_t_9-6-3m_b4','bh_ip_snf_admit_days_pmpm_t_9-6-3m_b4',
               'rx_phar_cat_cvs_pmpm_ct_t_9-6-3m_b4','total_physician_office_visit_ct_pmpm_t_6-3-0m_b4',
               'rx_generic_dist_gpi6_pmpm_ct_t_9-6-3m_b4','race_cd']


ignore_list = ['ID','cons_ltmedicr','mabh_seg','lang_spoken_cd','atlas_berry_acrespth12',
               'hedis_dia_hba1c_ge9','atlas_ghveg_sqftpth12']

interval_list = []

for c in holdout_clean_full.columns:
    if c not in nom_list:
        if c not in bn_list:
            if c not in ignore_list:
                interval_list.append(c)

holdout_clean_full[interval_list] = SimpleImputer(verbose=1).fit_transform(holdout_clean_full[interval_list])


for n in nom_list:
    holdout_clean_full[n] = holdout_clean_full[n].fillna(value=holdout_clean_full[n].mode()[0])
    
for b in bn_list:
    holdout_clean_full[b] = holdout_clean_full[b].fillna(value=holdout_clean_full[b].mode()[0])
    


holdout_clean_full = pd.get_dummies(holdout_clean_full,columns=nom_list)
holdout_clean_full = pd.get_dummies(holdout_clean_full,columns=bn_list,drop_first=True)

holdout_clean_full.to_pickle(folder+'Humana Holdout Data Imputed and Encoded.pkl')

cols = holdout_clean_full.columns.tolist()

encoded_df = holdout_clean_full.drop(['ID','cons_ltmedicr','mabh_seg','lang_spoken_cd','atlas_berry_acrespth12',
               'hedis_dia_hba1c_ge9','atlas_ghveg_sqftpth12'],axis=1)


encoded_df.to_csv(folder+'Humana Holdout Set Model-Ready.csv') 
