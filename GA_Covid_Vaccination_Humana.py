#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sara Foster
"""

from deap import creator, base, tools, algorithms

import random
import time
import pandas as pd
import numpy  as np
import matplotlib.pyplot               as plt
import statsmodels.api                 as sm
import statsmodels.tools.eval_measures as em
from AdvancedAnalytics.ReplaceImputeEncode import ReplaceImputeEncode, DT
from AdvancedAnalytics.Regression          import linreg, stepwise
from math                    import log, isfinite, sqrt, pi
from sklearn.linear_model    import LinearRegression, Lasso
from sklearn.metrics         import mean_squared_error, r2_score
from scipy.linalg            import qr_multiply, solve_triangular
              
def rng(z):
    r = maxBIC(z) - minBIC(z)
    return r

def avgBIC(z):
    sum = 0.0
    cnt = 0
    for i in range(len(z)):
        if isfinite(z[i][0]):
            sum += z[i][0]
            cnt += 1
    if cnt>0:
        return round(sum/cnt, 4)
    else:
        return np.nan

def maxBIC(z):
    max = 0
    for i in range(len(z)):
        if z[i][0] > max:
            max = z[i][0]
    return round(max, 3)

def minBIC(z):
    min = 1e64
    for i in range(len(z)):
        if z[i][0] < min:
            min = z[i][0]
    return round(min, 3)

def cvBIC(z):
    avg = avgBIC(z)
    std = stdBIC(z)
    if isfinite(avg):
        return round(100*std/avg, 3)
    else:
        return np.nan

def lbic(z):
    try:
        return round(log(minBIC(z)), 4)
    except:
        return 1e64

def stdBIC(z):
    sum   = 0.0
    sum2  = 0.0
    cnt   = 0
    for i in range(len(z)):
        if isfinite(z[i][0]):
            sum  += z[i][0]
            sum2 += z[i][0] * z[i][0]
            cnt += 1
    if cnt < 2:
        return np.nan
    else:
        sumsq = (sum*sum)/cnt
        return round(sqrt((sum2 - sumsq)/(cnt-1)), 4)
def features_(z):
    min     = 1e64
    feature = 1e64
    for i in range(len(z)):
        if z[i][0] < min:
            min = z[i][0]
            feature = z[i][1]
        if z[i][0] == min and z[i][1] < feature:
            feature = z[i][1]
    return feature

def geneticAlgorithm(X, y, n_population, n_generation, method='random',
                     n_int=None, n_nom=None, n_frac=None):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
 # create individual fitness dictionary
    ifit = {}
    # create individual
    creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("population_guess", initPopulation, list, 
                                                      creator.Individual)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                                     toolbox.attr_bool, n=len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
                                                      toolbox.individual)
    toolbox.register("evaluate", evalFitness, X=X, y=y, ifit=ifit)
    toolbox.register("mate",     tools.cxTwoPoint)
    toolbox.register("mutate",   tools.mutFlipBit, indpb=0.02)
    toolbox.register("select",   tools.selTournament, tournsize=7)

    if method=='random':
        pop   = toolbox.population(n_population)
    else:
        # initialize parameters
        # n_int Total number of interval features
        # n_nom List of number of dummy variables for each categorical var
        pop   = toolbox.population_guess(method, n_int, n_nom, n_frac)
        #n_population = len(pop)
    hof   = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", avgBIC)
    stats.register("std", stdBIC)
    stats.register("cv", cvBIC)
    stats.register("min", minBIC)
    stats.register("max", maxBIC)
    stats.register("range", rng)
    stats.register("Ln(BIC)", lbic)
    stats.register("Features", features_)

    # genetic algorithm
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.5,
                                   ngen=n_generation, stats=stats, 
                                   halloffame=hof, verbose=True)

    # return hall of fame
    return hof, logbook

def evalFitness(individual, X, y, ifit):
            
    cols  = [index for index in range(len(individual)) 
            if individual[index] != 1 ]# get features subset, drop features with cols[i] != 1
        
    X_selected = X.drop(X.columns[cols], axis=1)
    n = X_selected.shape[0]
    p = X_selected.shape[1]
    features = p
    ind = ""     
    for i in range(len(individual)):
        if individual[i] == 0:
            ind += '0'
        else:
            ind += '1'
    try:
        bic = ifit[ind]
        return(bic, features)
    except:
        pass
    
    abest = "bic"
    afit  = "statsmodels"
    afit  = "QR_decomp"
    if abest=="bic":
        boundary = 1e64
    else:
        boundary = 0
    if features > n-2:
        return (boundary, features)
    if afit == "QR_decomp":
        Xc       = sm.add_constant(X_selected)
        qty, r = qr_multiply(Xc, y)
        coef   = solve_triangular(r, qty)
        resid  = (Xc @ coef) - y
        ase    = (resid @ resid) / n
        if ase > 0:
            twoLL  = n*(log(2*pi) + 1.0 + log(ase))
            bic    = twoLL + log(n) * (Xc.shape[1]+1)
        else:
            bic  = -np.inf
        return(bic, features)
    elif afit == "statsmodels":
        Xc       = sm.add_constant(X_selected)
        model    = sm.OLS(y, Xc)
        results  = model.fit()
        parms    = np.ravel(results.params)
        if abest !="bic":
            pred  = model.predict(parms)
            R2    = r2_score(y, pred)
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            adjr2 = 1.0 - adjr2
            return(adjr2, features)
        else:
            loglike  = model.loglike(results.params)
            model_df = model.df_model + 2 #plus intercept and sigma
            nobs     = y.shape[0]
            bic      = em.bic(loglike, nobs, model_df)
            #aic      = em.aic(loglike, nobs, model_df)
            return(bic, features)
        
    else:
        lr    = LinearRegression().fit(X_selected,y)
        pred  = lr.predict(X_selected)
        k     = X_selected.shape[1] + 2
        ASE   = mean_squared_error(y,pred)
        twoLL = n*(log(2*pi) + 1.0 + log(ASE))
        bic   = twoLL + log(n)*k
        R2 = r2_score(y, pred)
        if R2 > 0.99999:
            bic = -np.inf
        return (bic, features)
        
def findBest(hof, X, y):
    #Find Best Individual in Hall of Fame
    print("Individuals in HoF: ", len(hof))
    bestBIC  = 1e64
    features = len(hof[0])
    for individual in hof:
        if(individual.fitness.values[0] < bestBIC):
            bestBIC = individual.fitness.values[0]
            _individual = individual
        if (sum(individual) < features and 
            individual.fitness.values[0] == bestBIC):
            features = sum(individual)
            _individual = individual

    _individualHeader = [list(X)[i] for i in range(len(_individual)) 
                        if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader

    
def initPopulation(pcls, ind_init, method, 
                   n_int, n_nom, n_frac):
    #k = number of columns in X
    #k1= number of interval variables (first k1 columns)
    #k2= number of other columns in X
    #k3= number of screening design cases (rows)
    k1 = n_int
    k2 = X.shape[1] - k1
    k = k1+k2
    # Initialize Null Case (no features)
    icls = [0]*k
    ind  = ind_init(icls)
    pcls = [ind]
    
    if method == 'star':
        # Add "All" one-feature selection (star points)
        for i in range(k):
            icls = [0]*k
            icls[i]  = 1
            ind = ind_init(icls)
            pcls.append(ind)
            
    return pcls
        
def plotGenerations(gen, lnbic, features):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("GA GENERATION", fontsize="x-large",fontweight="heavy")
    ax1.tick_params(axis='x', labelcolor="black", labelsize="x-large")
    ax1.tick_params(axis='y', labelcolor="green", labelsize="x-large")
    # Tick marks for the base axis of generation numbers.
    #ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax1.set_ylabel("Log(BIC)", fontsize="x-large", fontweight="heavy", color="green")
    ax1.set_facecolor((0.95,0.95,0.95))
    #ax1.grid(axis='x', linestyle='--', linewidth=1, color='gray')
    ax1.plot(gen, lnbic, 'go-', color="green", 
                         linewidth=2, markersize=10)
    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor="blue", labelsize="x-large")
    ax2.set_ylabel("Number of Features Selected", fontsize="x-large", 
                   fontweight="heavy", color="blue")
    # Tick marks for the right, vertical axis, the number of features
    #ax2.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    #ax2.grid(axis='y', linestyle='--', linewidth=1, color='gray')
    ax2.plot(gen, features, 'bs-', color="blue", 
                         linewidth=2, markersize=10)
    plt.savefig("randomStart.pdf")
    plt.show()
    
print("**********************************************************************")
data_map = {
	'ID': 		                              [ DT.Ignore   ,  ],
	'rx_gpi2_72_pmpm_cost_6to9m_b4': 	      [ DT.Interval , (-0.5, 3842.7168) ],
	'atlas_pct_laccess_child15': 	              [ DT.Interval , (-0.5, 31.656) ],
	'atlas_recfacpth14': 	                      [ DT.Interval , (-0.5, 1.1106) ],
	'atlas_pct_fmrkt_frveg16': 	              [ DT.Interval , (-0.5, 100.5) ],
	'atlas_pct_free_lunch14': 	              [ DT.Interval , (-0.4903, 100.5) ],
	'bh_ncal_pmpm_ct': 	                      [ DT.Interval , (-0.5, 5.8333) ],
	'src_div_id': 	                              [ DT.Nominal  , (0.0, 1.0, 2.0, 3.0, 4.0, 5.0) ],
	'total_bh_copay_pmpm_cost_t_9-6-3m_b4':       [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'cons_chmi': 	                              [ DT.Interval , (-0.5, 255.5) ],
	'mcc_ano_pmpm_ct_t_9-6-3m_b4': 	              [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Inc_1x-2x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'rx_maint_pmpm_cost_t_12-9-6m_b4': 	      [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
#	'cons_ltmedicr': 	                      [ DT.Ignore   , (-0.5, 9.5) ],
	'rx_gpi4_6110_pmpm_ct': 	              [ DT.Interval , (-0.5, 1.5833) ],
	'atlas_pc_snapben15': 	                      [ DT.Interval , (1.5808, 80.4416) ],
	'credit_bal_nonmtgcredit_60dpd': 	      [ DT.Interval , (78.38, 14805.3446) ],
	'rx_bh_mbr_resp_pmpm_cost_9to12m_b4':         [ DT.Interval , (-0.5, 542.75) ],
	'rx_nonbh_pmpm_cost_t_9-6-3m_b4': 	      [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'atlas_pct_laccess_nhna15': 	              [ DT.Interval , (-0.5, 58.3662) ],
	'credit_hh_nonmtgcredit_60dpd': 	      [ DT.Interval , (1.5351, 66.7352) ],
	'rx_bh_pmpm_ct_0to3m_b4': 	              [ DT.Interval , (-0.5, 7.5) ],
	'cons_lwcm10': 	                              [ DT.Interval , (-0.4741, 1.3209) ],
	'atlas_fsrpth14': 	                      [ DT.Interval , (-0.5, 4.8685) ],
	'auth_3mth_dc_home': 	                      [ DT.Interval , (0.0, 3.0) ],
	'atlas_wicspth12': 	                      [ DT.Interval , (-0.5, 2.2331) ],
	'rx_gpi2_17_pmpm_cost_t_12-9-6m_b4':          [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'cons_hxmioc': 	                              [ DT.Interval , (-0.5, 9.5) ],
	'rx_generic_pmpm_cost_t_6-3-0m_b4':           [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'cmsd2_sns_digest_abdomen_pmpm_ct':           [ DT.Interval , (-0.5, 10.125) ],
	'atlas_ghveg_farms12': 	                      [ DT.Interval , (-0.5, 150.5) ],
	'credit_hh_bankcardcredit_60dpd': 	      [ DT.Interval , (-0.46, 31.8409) ],
	'total_outpatient_allowed_pmpm_cost_6to9m_b4':[ DT.Interval , (-0.5, 26247.13) ],
	'cons_cwht': 	                              [ DT.Interval , (-0.5, 99.5) ],
	'atlas_netmigrationrate1016': 	              [ DT.Interval , (-16.2152, 43.478) ],
	'atlas_pct_laccess_snap15': 	              [ DT.Interval , (-0.5, 29.2418) ],
	'bh_ncdm_ind': 	                              [ DT.Binary   , (0, 1) ],
	'rx_nonmaint_mbr_resp_pmpm_cost_9to12m_b4':   [ DT.Interval , (-0.5, 1453.42) ],
	'atlas_retirement_destination_2015_upda':     [ DT.Binary   , (0.0, 1.0) ],
	'rx_overall_mbr_resp_pmpm_cost_t_6-3-0m_b4':  [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'atlas_naturalchangerate1016': 	              [ DT.Interval , (-7.0106, 14.8181) ],
	'ccsp_236_pct': 	                      [ DT.Interval , (-0.5, 11.9221) ],
	'rx_overall_dist_gpi6_pmpm_ct_t_6-3-0m_b4':   [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'atlas_pct_laccess_hisp15': 	              [ DT.Interval , (-0.5, 63.1906) ],
	'auth_3mth_dc_no_ref': 	                      [ DT.Interval , (0.0, 1.0) ],
	'rx_overall_mbr_resp_pmpm_cost': 	      [ DT.Interval , (-0.5, 1678.3184) ],
	'rx_overall_gpi_pmpm_ct_0to3m_b4': 	      [ DT.Interval , (-0.5, 42.1667) ],
	'auth_3mth_dc_snf': 	                      [ DT.Interval , (0.0, 2.0) ],
	'rx_phar_cat_humana_pmpm_ct_t_9-6-3m_b4':     [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'atlas_pct_laccess_hhnv15': 	              [ DT.Interval , (-0.5, 46.4558) ],
	'auth_3mth_acute_end': 	                      [ DT.Interval , (0, 1) ],
	'auth_3mth_psychic': 	                      [ DT.Interval , (0.0, 1.0) ],
	'atlas_hiamenity': 	                      [ DT.Binary   , (0.0, 1.0) ],
	'auth_3mth_bh_acute': 	                      [ DT.Interval , (0.0, 1.0) ],
	'credit_bal_consumerfinance': 	              [ DT.Interval , (89.3571, 5341.1712) ],
	'auth_3mth_acute_chf': 	                      [ DT.Interval , (0.0, 1.0) ],
	'rx_overall_gpi_pmpm_ct_t_6-3-0m_b4': 	      [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'rwjf_uninsured_pct': 	                      [ DT.Interval , (-0.4787, 0.8302) ],
	'mcc_chf_pmpm_ct_t_9-6-3m_b4': 	              [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'rx_mail_mbr_resp_pmpm_cost_0to3m_b4': 	      [ DT.Interval , (-0.5, 1218.79) ],
	'bh_urgent_care_copay_pmpm_cost_t_12-9-6m_b4':[ DT.Nominal  , ('No Activity', 'No_Change') ],
	'atlas_pct_wic15': 	                      [ DT.Interval , (0.6051, 3.7316) ],
	'ccsp_193_pct': 	                      [ DT.Interval , (-0.5, 8.5) ],
	'auth_3mth_dc_hospice': 	              [ DT.Interval , (0, 1) ],
	'atlas_pct_fmrkt_baked16': 	              [ DT.Interval , (-0.5, 100.5) ],
	'rx_nonmaint_mbr_resp_pmpm_cost':    	      [ DT.Interval , (-0.5, 1336.7341) ],
	'auth_3mth_acute_skn': 	                      [ DT.Interval , (0.0, 1.0) ],
	'atlas_veg_farms12': 	                      [ DT.Interval , (-0.5, 815.5) ],
	'atlas_vlfoodsec_13_15': 	              [ DT.Interval , (2.4, 8.4) ],
	'rx_gpi2_34_dist_gpi6_pmpm_ct': 	      [ DT.Interval , (-0.5, 1.5) ],
	'bh_ip_snf_net_paid_pmpm_cost': 	      [ DT.Interval , (-0.5, 1180.7633) ],
	'credit_hh_bankcard_severederog': 	      [ DT.Interval , (-0.4878, 27.3696) ],
	'rx_hum_16_pmpm_ct': 	                      [ DT.Interval , (-0.5, 12.0) ],
	'est_age': 	                              [ DT.Interval , (19.5, 104.5) ],
	'rx_maint_pmpm_cost_t_6-3-0m_b4': 	      [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'cnt_cp_webstatement_pmpm_ct':                [ DT.Interval , (-0.5, 41.0) ],
	'atlas_pct_laccess_seniors15': 	              [ DT.Interval , (-0.5, 28.4463) ],
	'phy_em_px_pct': 	                      [ DT.Interval , (-0.5, 12.5) ],
	'atlas_percapitainc': 	                      [ DT.Interval , (7123.5, 66522.5) ],
	'rwjf_uninsured_adults_pct': 	              [ DT.Interval , (-0.4738, 0.934) ],
	'rx_generic_mbr_resp_pmpm_cost_0to3m_b4':     [ DT.Interval , (-0.5, 671.2067) ],
	'rwjf_air_pollute_density': 	              [ DT.Interval , (4.0, 15.9) ],
	'rx_gpi2_02_pmpm_cost': 	              [ DT.Interval , (-0.5, 100.5) ],
	'atlas_recfac14': 	                      [ DT.Interval , (-0.5, 845.5) ],
	'cons_mobplus': 	                      [ DT.Nominal  , ('M', 'P', 'S', 'U') ],
	'lab_albumin_loinc_pmpm_ct': 	              [ DT.Interval , (-0.5, 1.5) ],
	'atlas_pct_obese_adults13': 	              [ DT.Interval , (11.3, 46.8) ],
	'rx_maint_net_paid_pmpm_cost_t_12-9-6m_b4':   [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'rev_pm_obsrm_pmpm_ct': 	              [ DT.Interval , (-0.5, 3.0) ],
	'atlas_pct_sfsp15': 	                      [ DT.Interval , (-0.2729, 4.7713) ],
	'total_physician_office_net_paid_pmpm_cost_9to12m_b4': 	[ DT.Interval , (-0.5, 541.475) ],
	'atlas_pc_dirsales12': 	                                [ DT.Interval , (-0.5, 126.9246) ],
	'med_ip_snf_admit_days_pmpm': 	                        [ DT.Interval , (-0.5, 13.1667) ],
	'rej_med_outpatient_visit_ct_pmpm_t_6-3-0m_b4': 	[ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'cms_tot_partd_payment_amt': 	                        [ DT.Interval , (-0.5, 813.56) ],
	'rx_nonotc_dist_gpi6_pmpm_ct': 	                        [ DT.Interval , (-0.5, 13.5) ],
	'rx_nonmaint_pmpm_ct': 	                                [ DT.Interval , (-0.5, 18.0) ],
	'rx_nonbh_mbr_resp_pmpm_cost_6to9m_b4': 	        [ DT.Interval , (-0.5, 2233.0632) ],
	'cons_stlnindx': 	                                [ DT.Interval , (-0.5, 9.5) ],
	'atlas_hipov_1115': 	                                [ DT.Binary   , (0.0, 1.0) ],
	'rx_nonbh_mbr_resp_pmpm_cost': 	                        [ DT.Interval , (-0.5, 1651.5841) ],
	'atlas_redemp_snaps16': 	                        [ DT.Interval , (0.4989, 827698.2406) ],
	'atlas_berry_farms12': 	                                [ DT.Interval , (-0.5, 360.5) ],
	'rej_med_ip_snf_coins_pmpm_cost_t_9-6-3m_b4': 	        [ DT.Nominal  , ('No Activity', 'No_Change') ],
	'rwjf_inactivity_pct': 	                                [ DT.Interval , (-0.402, 0.944) ],
	'rx_gpi2_72_pmpm_ct_6to9m_b4': 	                        [ DT.Interval , (-0.5, 4.5) ],
	'cons_n2pmr': 	                                        [ DT.Interval , (-0.5, 92.5) ],
	'med_physician_office_allowed_pmpm_cost_t_9-6-3m_b4': 	[ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'auth_3mth_acute_res': 	                                [ DT.Interval , (0.0, 1.0) ],
	'rev_cms_ct_pmpm_ct': 	                                [ DT.Interval , (-0.5, 1.8333) ],
	'atlas_foodhub16': 	                                [ DT.Interval , (0.0, 6.0) ],
	'total_physician_office_copay_pmpm_cost': 	        [ DT.Interval , (-0.5, 118.7342) ],
	'auth_3mth_acute_dig': 	                                [ DT.Interval , (0.0, 1.0) ],
	'auth_3mth_dc_acute_rehab': 	                        [ DT.Interval , (0.0, 1.0) ],
	'atlas_pct_fmrkt_anmlprod16': 	                        [ DT.Interval , (-0.5, 100.5) ],
	'credit_num_agencyfirstmtg': 	                        [ DT.Interval , (-0.5, 1.2297) ],
	'total_physician_office_net_paid_pmpm_cost_t_9-6-3m_b4':[ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'atlas_type_2015_mining_no': 	                        [ DT.Binary   , (0.0, 1.0) ],
	'atlas_agritrsm_rct12': 	                        [ DT.Interval , (-0.5, 6899000.5) ],
	'rx_days_since_last_script': 	                        [ DT.Interval , (0.5, 480.5) ],
	'atlas_pct_laccess_pop15': 	                        [ DT.Interval , (-0.5, 100.5) ],
	'auth_3mth_post_acute_res': 	                        [ DT.Interval , (0, 1) ],
	'auth_3mth_acute_inf': 	                                [ DT.Interval , (0, 1) ],
	'rx_gpi2_01_pmpm_cost_0to3m_b4': 	                [ DT.Interval , (-0.5, 352.55) ],
	'atlas_povertyallagespct': 	                        [ DT.Interval , (2.9, 45.7) ],
	'covid_vaccination': 	                                [ DT.Binary   , ('no_vacc', 'vacc') ],
	'rwjf_uninsured_child_pct': 	                        [ DT.Interval , (-0.4917, 0.743) ],
	'rx_branded_pmpm_ct_t_6-3-0m_b4': 	                [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'med_outpatient_deduct_pmpm_cost_t_9-6-3m_b4': 	        [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'credit_bal_mtgcredit_new': 	                        [ DT.Interval , (-0.4907, 95674.1707) ],
	'atlas_low_employment_2015_update': 	                [ DT.Binary   , (0.0, 1.0) ],
	'atlas_pct_diabetes_adults13': 	                        [ DT.Interval , (2.8, 22.2) ],
	'atlas_pct_laccess_nhasian15': 	                        [ DT.Interval , (-0.5, 10.4636) ],
	'atlas_deep_pov_all': 	                                [ DT.Interval , (0.5672, 38.6865) ],
	'atlas_net_international_migration_rate': 	        [ DT.Interval , (-0.9028, 10.1939) ],
	'atlas_deep_pov_children': 	                        [ DT.Interval , (-0.4998, 55.5641) ],
	'bh_ncdm_pct': 	                                        [ DT.Interval , (-0.5, 12.5) ],
	'auth_3mth_non_er': 	                                [ DT.Interval , (0.0, 2.0) ],
	'atlas_foodinsec_child_03_11': 	                        [ DT.Interval , (4.6, 13.3) ],
	'rx_branded_mbr_resp_pmpm_cost': 	                [ DT.Interval , (-0.5, 1649.1575) ],
	'atlas_pc_wic_redemp12': 	                        [ DT.Interval , (-0.3606, 94.0552) ],
	'rwjf_mv_deaths_rate': 	                                [ DT.Interval , (2.4232, 73.9603) ],
	'auth_3mth_acute_cad': 	                                [ DT.Interval , (0, 1) ],
	'atlas_pct_reduced_lunch14': 	                        [ DT.Interval , (-0.5, 21.4477) ],
	'cons_nwperadult': 	                                [ DT.Interval , (-2500.5, 1000000.5) ],
	'total_allowed_pmpm_cost_t_9-6-3m_b4': 	                [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'rx_hum_28_pmpm_cost': 	                                [ DT.Interval , (-0.5, 199.6442) ],
#	'mabh_seg': 	                                        [ DT.Ignore   , ('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8') ],
	'cms_orig_reas_entitle_cd': 	                        [ DT.Nominal  , (0.0, 1.0, 2.0, 3.0) ],
	'atlas_totalocchu': 	                                [ DT.Interval , (313.5, 3281845.5) ],
	'med_physician_office_ds_clm_6to9m_b4': 	        [ DT.Interval , (119.5, 273.5) ],
	'atlas_pct_loclfarm12': 	                        [ DT.Interval , (-0.5, 80.5) ],
	'rx_generic_mbr_resp_pmpm_cost': 	                [ DT.Interval , (-0.5, 555.8275) ],
	'total_outpatient_mbr_resp_pmpm_cost_6to9m_b4': 	[ DT.Interval , (-0.5, 2780.59) ],
	'rx_gpi4_3400_pmpm_ct': 	                        [ DT.Interval , (-0.5, 4.5) ],
	'lab_dist_loinc_pmpm_ct': 	                        [ DT.Interval , (-0.5, 68.5) ],
	'atlas_pct_nslp15': 	                                [ DT.Interval , (6.1689, 13.3538) ],
	'rx_generic_pmpm_ct_0to3m_b4': 	                        [ DT.Interval , (-0.5, 39.5) ],
	'oontwk_mbr_resp_pmpm_cost_t_6-3-0m_b4': 	        [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'atlas_pct_laccess_lowi15': 	                        [ DT.Interval , (-0.5, 53.1667) ],
	'bh_ncal_ind': 	                                        [ DT.Binary   , (0, 1) ],
	'atlas_pct_fmrkt_sfmnp16': 	                        [ DT.Interval , (-0.5, 100.5) ],
	'hum_region': 	                                        [ DT.Nominal  , ('CALIFORNIA/NEVADA', 'CENTRAL', 'CENTRAL WEST', 'EAST', 'EAST CENTRAL', 'FLORIDA', 'GREAT LAKES/CENTRAL NORTH', 'GULF STATES', 'INTERMOUNTAIN', 'MID-ATLANTIC/NORTH CAROLINA', 'MID-SOUTH', 'NORTHEAST', 'PACIFIC', 'PR', 'SOUTHEAST', 'TEXAS') ],
	'rx_nonmail_dist_gpi6_pmpm_ct_t_9-6-3m_b4': 	        [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'atlas_pct_loclsale12': 	                        [ DT.Interval , (-0.5, 36.3025) ],
	'rej_med_er_net_paid_pmpm_cost_t_9-6-3m_b4': 	        [ DT.Nominal  , ('No Activity', 'No_Change') ],
	'credit_bal_autobank': 	                                [ DT.Interval , (673.7134, 16942.1012) ],
	'med_outpatient_mbr_resp_pmpm_cost_t_9-6-3m_b4':        [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'rx_overall_mbr_resp_pmpm_cost_0to3m_b4': 	        [ DT.Interval , (-0.5, 1754.45) ],
	'rx_tier_2_pmpm_ct_3to6m_b4': 	                        [ DT.Interval , (-0.5, 19.8333) ],
	'rx_nonbh_net_paid_pmpm_cost': 	                        [ DT.Interval , (-0.5, 39533.426) ],
	'rx_maint_pmpm_ct_9to12m_b4': 	                        [ DT.Interval , (-0.5, 36.8333) ],
	'rx_nonbh_net_paid_pmpm_cost_t_6-3-0m_b4':  	        [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'atlas_type_2015_recreation_no': 	                [ DT.Binary   , (0.0, 1.0) ],
	'rx_gpi2_39_pmpm_cost_t_6-3-0m_b4': 	                [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'atlas_type_2015_update': 	                        [ DT.Nominal  , (0.0, 1.0, 2.0, 3.0, 4.0, 5.0) ],
	'cms_risk_adjustment_factor_a_amt': 	                [ DT.Interval , (-0.5, 10.772) ],
	'total_ip_maternity_net_paid_pmpm_cost_t_12-9-6m_b4': 	[ DT.Nominal  , ('No Activity', 'No_Change') ],
	'rx_generic_pmpm_cost': 	                        [ DT.Interval , (-0.5, 3718.7524) ],
	'cmsd2_eye_retina_pmpm_ct': 	                        [ DT.Interval , (-0.5, 2.6667) ],
	'auth_3mth_post_acute': 	                        [ DT.Interval , (0.0, 2.0) ],
	'auth_3mth_facility': 	                                [ DT.Interval , (0.0, 2.0) ],
	'rx_days_since_last_script_0to3m_b4': 	                [ DT.Interval , (0.5, 120.5) ],
	'atlas_population_loss_2015_update': 	                [ DT.Binary   , (0.0, 1.0) ],
	'rx_maint_pmpm_ct_t_6-3-0m_b4': 	                [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'auth_3mth_acute_mean_los': 	                        [ DT.Interval , (-0.5, 42.5) ],
	'credit_num_autofinance': 	                        [ DT.Interval , (-0.4562, 1.8466) ],
	'cons_rxmaint': 	                                [ DT.Interval , (-0.5, 9.5) ],
	'rx_mail_net_paid_pmpm_cost_t_6-3-0m_b4': 	        [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'auth_3mth_home': 	                                [ DT.Interval , (0.0, 3.0) ],
	'rx_maint_mbr_resp_pmpm_cost_6to9m_b4': 	        [ DT.Interval , (-0.5, 1688.26) ],
	'cons_hxwearbl': 	                                [ DT.Interval , (-0.5, 9.5) ],
	'total_physician_office_mbr_resp_pmpm_cost_t_9-6-3m_b4':[ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'atlas_pct_laccess_black15': 	                        [ DT.Interval , (-0.5, 38.0687) ],
	'atlas_hh65plusalonepct': 	                        [ DT.Interval , (2.8088, 24.4709) ],
	'atlas_farm_to_school13': 	                        [ DT.Binary   , (0.0, 1.0) ],
	'auth_3mth_acute_inj': 	                                [ DT.Interval , (0, 1) ],
	'rej_days_since_last_clm': 	                        [ DT.Interval , (2.5, 480.5) ],
	'bh_outpatient_net_paid_pmpm_cost': 	                [ DT.Interval , (-0.5, 1287.05) ],
	'atlas_dirsales_farms12': 	                        [ DT.Interval , (-0.5, 839.5) ],
	'rx_generic_pmpm_cost_6to9m_b4': 	                [ DT.Interval , (-0.5, 4619.73) ],
	'rev_cms_ansth_pmpm_ct': 	                        [ DT.Interval , (-0.5, 1.0) ],
	'atlas_convspth14': 	                                [ DT.Interval , (-0.4999, 3.2462) ],
	'total_med_allowed_pmpm_cost_9to12m_b4': 	        [ DT.Interval , (-0.5, 65189.07) ],
	'rx_mail_mbr_resp_pmpm_cost_t_9-6-3m_b4': 	        [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'med_outpatient_visit_ct_pmpm_t_12-9-6m_b4': 	        [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'rx_nonbh_pmpm_ct_t_9-6-3m_b4': 	                [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'auth_3mth_acute': 	                                [ DT.Interval , (0.0, 4.0) ],
	'rx_nonbh_pmpm_ct_0to3m_b4': 	                        [ DT.Interval , (-0.5, 35.5) ],
	'atlas_pc_ffrsales12': 	                                [ DT.Interval , (363.612, 1035.8916) ],
	'auth_3mth_dc_left_ama': 	                        [ DT.Interval , (0.0, 1.0) ],
	'credit_bal_bankcard_severederog': 	                [ DT.Interval , (-0.4998, 4787.7302) ],
	'atlas_povertyunder18pct': 	                        [ DT.Interval , (2.4, 66.8) ],
	'rx_tier_1_pmpm_ct_0to3m_b4': 	                        [ DT.Interval , (-0.5, 22.1667) ],
	'cons_estinv30_rc': 	                                [ DT.Interval , (2499.5, 1000000.5) ],
	'auth_3mth_bh_acute_men': 	                        [ DT.Interval , (0, 1) ],
	'rx_gpi2_34_pmpm_ct': 	                                [ DT.Interval , (-0.5, 4.5) ],
	'auth_3mth_dc_custodial': 	                        [ DT.Interval , (0.0, 1.0) ],
	'atlas_veg_acrespth12': 	                        [ DT.Interval , (-0.5, 2600.7403) ],
	'atlas_grocpth14': 	                                [ DT.Interval , (-0.5, 2.1155) ],
	'total_med_net_paid_pmpm_cost_t_6-3-0m_b4': 	        [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'rx_gpi2_90_dist_gpi6_pmpm_ct_9to12m_b4': 	        [ DT.Interval , (0, 1.5)],
	'atlas_csa12': 	                                        [ DT.Interval , (-0.5, 88.5) ],
	'sex_cd': 	                                        [ DT.Binary   , ('F', 'M') ],
	'rx_gpi2_62_pmpm_cost_t_9-6-3m_b4': 	                [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
#	'lang_spoken_cd': 	                                [ DT.Ignore   , ('CHI', 'CRE', 'ENG', 'KOR', 'OTH', 'SPA', 'VIE') ],
	'rx_overall_gpi_pmpm_ct_t_12-9-6m_b4': 	                [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'cons_hhcomp': 	                                        [ DT.Nominal  , ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'U') ],
	'auth_3mth_acute_hdz': 	                                [ DT.Interval , (0, 1) ],
	'cons_rxadhs': 	                                        [ DT.Interval , (-0.5, 9.5) ],
	'atlas_pct_fmrkt_snap16': 	                        [ DT.Interval , (-0.5, 100.5) ],
	'met_obe_diag_pct': 	                                [ DT.Interval , (-0.5, 12.5) ],
	'cms_partd_ra_factor_amt': 	                        [ DT.Interval , (-0.5, 12.214) ],
	'atlas_pct_sbp15':                                     	[ DT.Interval , (1.0464, 8.6602) ],
	'rwjf_resident_seg_black_inx': 	                        [ DT.Interval , (-0.2416, 90.8656) ],
	'atlas_pct_cacfp15': 	                                [ DT.Interval , (0.0681, 2.8593) ],
	'auth_3mth_rehab': 	                                [ DT.Interval , (0, 1) ],
	'pdc_lip': 	                                        [ DT.Interval , (-0.3352, 1.6) ],
	'atlas_ffrpth14': 	                                [ DT.Interval , (-0.5, 2.7812) ],
	'credit_num_autobank_new': 	                        [ DT.Interval , (-0.4999, 0.7981) ],
	'rx_tier_2_pmpm_ct': 	                                [ DT.Interval , (-0.5, 18.4167) ],
	'cons_n2pwh': 	                                        [ DT.Interval , (-0.5, 99.5) ],
	'rx_nonmaint_dist_gpi6_pmpm_ct_t_12-9-6m_b4': 	        [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
#	'atlas_berry_acrespth12': 	                        [ DT.Ignore   , (-0.5, 61.401) ],
	'atlas_pct_fmrkt_credit16': 	                        [ DT.Interval , (-0.5, 100.5) ],
	'atlas_slhouse12': 	                                [ DT.Interval , (-0.5, 23.5) ],
	'atlas_pc_fsrsales12': 	                                [ DT.Interval , (439.2315, 2161.0031) ],
	'credit_hh_1stmtgcredit': 	                        [ DT.Interval , (0.8176, 71.2109) ],
	'auth_3mth_snf_post_hsp': 	                        [ DT.Interval , (0.0, 2.0) ],
	'atlas_pct_fmrkt_wiccash16': 	                        [ DT.Interval , (-0.5, 100.5) ],
	'atlas_foodinsec_13_15': 	                        [ DT.Interval , (8.0, 21.3) ],
	'cons_rxadhm': 	                                        [ DT.Interval , (-0.5, 9.5) ],
	'atlas_fmrktpth16': 	                                [ DT.Interval , (-0.5, 1.2396) ],
	'rx_nonotc_pmpm_cost_t_6-3-0m_b4': 	                [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'cci_dia_m_pmpm_ct': 	                                [ DT.Interval , (-0.5, 15.8333) ],
	'cons_n2phi': 	                                        [ DT.Interval , (-0.5, 99.5) ],
	'bh_physician_office_copay_pmpm_cost_6to9m_b4': 	[ DT.Interval , (-0.5, 57.78) ],
	'rwjf_income_inequ_ratio': 	                        [ DT.Interval , (2.47, 9.429) ],
	'rej_total_physician_office_visit_ct_pmpm_0to3m_b4': 	[ DT.Interval , (0.0, 2.0) ],
	'credit_num_nonmtgcredit_60dpd': 	                [ DT.Interval , (-0.475, 2.6726) ],
	'credit_bal_autofinance_new': 	                        [ DT.Interval , (3.004, 7448.1231) ],
	'rwjf_men_hlth_prov_ratio': 	                        [ DT.Interval , (-0.5, 0.51) ],
	'auth_3mth_dc_home_health': 	                        [ DT.Interval , (0.0, 2.0) ],
	'rx_gpi2_56_dist_gpi6_pmpm_ct_3to6m_b4': 	        [ DT.Interval , (0.0, 1.0) ],
	'cmsd2_sns_genitourinary_pmpm_ct': 	                [ DT.Interval , (-0.5, 6.3333) ],
	'auth_3mth_acute_cir': 	                                [ DT.Interval , (0, 1) ],
	'med_ambulance_coins_pmpm_cost_t_9-6-3m_b4': 	        [ DT.Nominal  , ('Inc_1x-2x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
#	'hedis_dia_hba1c_ge9': 	                                [ DT.Ignore   , ('N', 'Y') ],
	'bh_ncal_pct': 	                                        [ DT.Interval , (-0.5, 12.5) ],
	'atlas_pct_snap16': 	                                [ DT.Interval , (5.3178, 23.2637) ],
	'ccsp_227_pct': 	                                [ DT.Interval , (-0.5, 12.5) ],
#	'atlas_ghveg_sqftpth12': 	                        [ DT.Ignore   , (-0.5, 11244.3574) ],
	'rx_days_since_last_script_6to9m_b4': 	                [ DT.Interval , (-0.5, 274.5) ],
	'atlas_orchard_acrespth12': 	                        [ DT.Interval , (-0.5, 900.5) ],
	'atlas_persistentchildpoverty_1980_2011': 	        [ DT.Binary   , (0.0, 1.0) ],
	'atlas_pct_laccess_multir15': 	                        [ DT.Interval , (-0.5, 16.553) ],
	'cons_cgqs': 	                                        [ DT.Interval , (59.5, 170.5) ],
	'ccsp_065_pmpm_ct': 	                                [ DT.Interval , (0.0, 0.18) ],
	'atlas_medhhinc': 	                                [ DT.Interval , (22044.5, 134609.5) ],
	'rx_maint_net_paid_pmpm_cost_t_9-6-3m_b4': 	        [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'rwjf_mental_distress_pct': 	                        [ DT.Interval , (-0.4197, 0.6937) ],
	'bh_ip_snf_admit_days_pmpm_t_9-6-3m_b4': 	        [ DT.Nominal  , ('No Activity', 'No_Change') ],
	'rx_phar_cat_cvs_pmpm_ct_t_9-6-3m_b4': 	                [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'zip_cd': 	                                        [ DT.Interval , (601.5, 99827.5) ],
	'atlas_pct_laccess_nhpi15': 	                        [ DT.Interval , (-0.5, 4.7699) ],
	'auth_3mth_post_er': 	                                [ DT.Interval , (0.0, 3.0) ],
	'credit_num_consumerfinance_new': 	                [ DT.Interval , (-0.4805, 1.4977) ],
	'rx_gpi2_49_pmpm_cost_0to3m_b4': 	                [ DT.Interval , (-0.5, 651.5933) ],
	'cons_chva': 	                                        [ DT.Interval , (-0.5, 999.5) ],
	'atlas_avghhsize': 	                                [ DT.Interval , (1.25, 4.38) ],
	'rx_overall_net_paid_pmpm_cost_6to9m_b4': 	        [ DT.Interval , (-0.5, 41736.332) ],
	'atlas_ownhomepct': 	                                [ DT.Interval , (18.5669, 92.8509) ],
	'atlas_orchard_farms12': 	                        [ DT.Interval , (-0.5, 4124.5) ],
	'total_physician_office_visit_ct_pmpm_t_6-3-0m_b4': 	[ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'atlas_pct_fmrkt_wic16': 	                        [ DT.Interval , (-0.5, 100.5) ],
	'rx_gpi2_33_pmpm_ct_0to3m_b4': 	                        [ DT.Interval , (0.0, 2.5) ],
	'rwjf_social_associate_rate': 	                        [ DT.Interval , (-0.4953, 49.1772) ],
	'atlas_freshveg_farms12': 	                        [ DT.Interval , (-0.5, 802.5) ],
	'auth_3mth_acute_sns': 	                                [ DT.Interval , (0, 1) ],
	'days_since_last_clm_0to3m_b4': 	                [ DT.Interval , (2.5, 120.5) ],
	'auth_3mth_dc_other': 	                                [ DT.Interval , (0.0, 1.0) ],
	'auth_3mth_bh_acute_mean_los': 	                        [ DT.Interval , (0.0, 5.0) ],
	'mcc_end_pct': 	                                        [ DT.Interval , (-0.5, 12.5) ],
	'cons_lwcm07': 	                                        [ DT.Interval , (-0.4628, 1.2202) ],
	'atlas_pct_fmrkt_otherfood16': 	                        [ DT.Interval , (-0.5, 100.5) ],
	'auth_3mth_acute_mus': 	                                [ DT.Interval , (0.0, 1.0) ],
	'atlas_perpov_1980_0711': 	                        [ DT.Binary   , (0.0, 1.0) ],
	'atlas_pct_laccess_white15': 	                        [ DT.Interval , (-0.5, 92.5787) ],
	'auth_3mth_post_acute_mean_los': 	                [ DT.Interval , (-0.5, 43.5) ],
	'rx_gpi2_66_pmpm_ct': 	                                [ DT.Interval , (-0.5, 3.2273) ],
	'auth_3mth_acute_gus':                                  [ DT.Interval , (0, 1) ],
	'rx_generic_dist_gpi6_pmpm_ct_t_9-6-3m_b4': 	        [ DT.Nominal  , ('Dec_1x-2x', 'Dec_2x-4x', 'Dec_4x-8x', 'Dec_over_8x', 'Inc_1x-2x', 'Inc_2x-4x', 'Inc_4x-8x', 'Inc_over_8x', 'New', 'No Activity', 'No_Change', 'Resolved') ],
	'atlas_low_education_2015_update': 	                [ DT.Binary   , (0.0, 1.0) ],
	'race_cd': 	                                        [ DT.Nominal  , (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0) ]
    }


target='covid_vaccination'
folder = 'C:/Users/foster-s/OneDrive - Texas A&M University/Python Projects/Humana/'
df = pd.read_pickle(folder+'training_clean_full.pickle')

df = df.replace('*',None)
df = df.replace(np.nan,None)

rie = ReplaceImputeEncode(data_map = data_map,binary_encoding='one-hot',
                          nominal_encoding='one-hot',no_impute=[target],
                          drop=True,display=True)

encoded_df = rie.fit_transform(df)
    
print("Read", df.shape[0], "observations with ", 
      df.shape[1], "attributes:\n")

y = encoded_df[target] # The target is not scaled or imputed
X = encoded_df.drop(target, axis=1)

print("************* GA SELECTION *******************")
# apply genetic algorithm
# n_int:   set to the number of candidate interval and binary features
# n_nom:   set to a list of levels for each candidate nominal feature
#          if there are no candidate nominal features, set to an empty list []
n_int = 239       # an integer 0 or greater
n_nom = [2, 4, 6, 9, 11, 13, 16] # 42 dummy features
p     = n_int + sum(n_nom) # Total number of features 52

# models:   the list of currently available statistical models
# fitness: the list of currently available fitness functions
# init:    the list of currently available initialization algorithms
#          each initialization algorithm can be used to initialize 
#          generation zero.  Select the one that produces a generation zero
#          closest to the imagined best number of features.  'star' starts 
#          with only one feature per individual.  'random' starts with a
#          larger number of features per individual, approximate half the
#          total number of candidates.
models     = [ 'sklearn', 'statsmodels', 'QR_decomp']
fitness    = ['bic', 'aic', 'AdjR2']
init       = ['star', 'random']
# Set calcModel, goodFit and initMethod to your choice for the statistical
#     model, the goodness of fit metric, and the initialization algorithm.
calcModel  = models [0]
goodFit    = fitness[0]
initMethod = init[0] 
# Initial generation has only 1 feature per individual.
# Initial generation with 'random' has about 50% of all features.
# n_pop is the initial population size.  Subsequent generations will be near
#       this size.
# n_gen is the number of generations, each progressively better than the 
#       previous generation.  This needs to be large enough to all the 
#       search algorithm to identify the best feature selection.
# Note: This algorithm optimizes the fitness of the individual while 
#       minimizing the number of features selected for the model.
if initMethod=='star':
    n_pop = p+1
    n_gen =  50
else:
    n_pop = 100
    n_gen =  50

print("{:*>71s}".format('*'))
print("{:*>14s}     GA Selection using {:>5s} Fitness         {:*>11s}". 
      format('*', goodFit, '*'))
print("{:*>14s} {:>11s} Models and {:>6s} Initialization {:*>11s}". 
      format('*', calcModel, initMethod, '*'))
print(" ")
random.seed(12345)
start = time.time()
hof, logbook = geneticAlgorithm(X, y, n_pop, n_gen, method=initMethod,
                                reg='linear', goodFit=goodFit,
                                calcModel=calcModel, n_int=n_int, n_nom=n_nom)

gen, features, min_, avg_, max_, rng_, lnfit = logbook.select("gen",
                    "features", "min", "avg", "max", "range", "Ln(Fit)")
end = time.time()    
duration = end-start
print("GA Runtime ", duration, " sec.")

# Plot Fitness and Number of Features versus Generation
plotGenerations(gen, lnfit, features)

# select the best individual
fit, individual, header = findBest(hof, goodFit, X, y)
print("Best Fitness:", fit[0])
print("Number of Features Selecterd: ", len(header))
print("\nFeatures:", header)

Xc = sm.add_constant(X[header])
model = sm.OLS(y, Xc)
results = model.fit()
print(results.summary())
logl  = model.loglike(results.params)
model_df = model.df_model + 2 #plus intercept and sigma
nobs     = y.shape[0]
bic      = em.bic(logl, nobs, model_df)
aic      = em.aic(logl, nobs, model_df)
print("BIC: ", bic)
print(" ")
print("{:*>71s}".format('*'))
print("{:*>14s}     STEPWISE SELECTION    {:*>30s}". format('*', '*'))
print("{:*>71s}".format('*'))

sw       = stepwise(encoded_df, target, reg="linear", method="stepwise",
                    crit_in=0.1, crit_out=0.1, verbose=True)
selected = sw.fit_transform()
print("Number of Selected Features: ", len(selected))
Xc  = sm.add_constant(encoded_df[selected])
model   = sm.OLS(y, Xc)
results = model.fit()            
print(results.summary())
logl  = model.loglike(results.params)
model_df = model.df_model + 2 #plus intercept and sigma
nobs     = y.shape[0]
bic      = em.bic(logl, nobs, model_df)
aic      = em.aic(logl, nobs, model_df)
print("BIC: ", bic)
print(" ")    
print("{:*>71s}".format('*'))
print("{:*>14s}     REPLACE IMPUTE ENCODE      {:*>25s}". format('*', '*'))
print("{:*>71s}".format('*'))

rie = ReplaceImputeEncode(data_map = data_map,binary_encoding='one-hot',
                          nominal_encoding='one-hot',no_impute=[target],
                          drop=True,display=True)
encoded_df = rie.fit_transform(df)

y = encoded_df[target]
X = encoded_df.drop(target, axis=1)
print(" ")
print("{:*>71s}".format('*'))
print("{:*>14s}     FIT FULL MODEL        {:*>30s}". format('*', '*'))
print("{:*>71s}".format('*'))
Xc = sm.add_constant(X)
lr = sm.OLS(y, Xc)
results = lr.fit()
print(results.summary())
ll       = lr.loglike(results.params)
model_df = lr.df_model + 2 #plus intercept and sigma
nobs     = y.shape[0]
bic      = em.bic(ll, nobs, model_df)
print("BIC:", bic)
print(" ")
print("{:*>71s}".format('*'))
#print("{:*>18s}        LASSO       {:*>33s}". format('*', '*'))
#print("{:*>71s}".format('*'))
#alpha_list = [0.00001, 0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 
#              0.006, 0.007, 0.008, 0.009, 0.01, 0.1, 1.0]
#for a in alpha_list:
#    clr = Lasso(alpha=a, random_state=12345)
#    clr.fit(X, y)
#    c = clr.coef_
#    z = 0
#    for i in range(len(c)):
#        if abs(c[i]) > 1e-3:
#            z = z+1
#    print("\nAlpha: ", a, " Number of Coefficients: ", z, "/", len(c))
#    linreg.display_metrics(clr, X, y)
#
#print("{:*>71s}".format('*'))
