# functions for analysis of tweets from reduced anonymised sets

import sys
import os
import time
import datetime as dt
import math
import numpy as np 
import scipy as sp
import pandas as pd


# facilitate counting of 
def ind_set(df,col_sorted,thresh1,thresh2):
    t1 = df[col_sorted].searchsorted(thresh1)
    t2 = df[col_sorted].searchsorted(thresh2)
    return df.loc[t1:t2-1]

def ind_set_counts(df,col_sort,bins):
	#bins is a series/array of stamps in the same value as that in sorted colum of df col_sorted
	df=df.sort_values(col_sort).reset_index(drop = True)
	df_counted = pd.DataFrame()
	df_counted['Bounds']= bins
	df_counted['Counts'] = 0
	for i in range(len(df_counted)-1):
		df_sub = ind_set(df,col_sort,bins[i],bins[i+1])
		df_counted.loc[i,'Counts']=len(df_sub) #or maybe it's 1?
	df_sub =df.loc[df[col_sort].searchsorted(bins[len(bins)-1]):]
	df_counted.loc[len(bins)-1,'Counts']=len(df_sub) #or maybe it's 1?
	return df_counted

def ind_set_unique_counts(df,col_sorted,col_unique,bins):
	#bins is a series/array of stamps in the same value as that in sorted colum of df col_sorted
	df=df.sort_values(col_sorted).reset_index(drop = True)
	df_counted = pd.DataFrame()
	df_counted['Bounds']= bins
	df_counted['Counts'] = 0
	for i in range(len(df_counted)-1):
		df_sub = ind_set(df,col_sorted,bins[i],bins[i+1])
		df_counted.loc[i,'Counts']=len(df_sub[col_unique].unique()) #or maybe it's 1?
	df_sub =df.loc[df[col_sorted].searchsorted(bins[len(bins)-1]):]
	df_counted.loc[len(bins)-1,'Counts']=len(df_sub[col_unique].unique()) #or maybe it's 1?
	return df_counted

def ind_set_total_counts(df,col_sorted,col_tot,bins):
	#bins is a series/array of stamps in the same value as that in sorted colum of df col_sorted
	df=df.sort_values(col_sorted).reset_index(drop = True)
	df_counted = pd.DataFrame()
	df_counted['Bounds']= bins
	df_counted['Sums'] = 0
	for i in range(len(df_counted)-1):
		df_sub = ind_set(df,col_sorted,bins[i],bins[i+1])
		df_counted.loc[i,'Sums']=df_sub[col_tot].sum() #or maybe it's 1?
	df_sub = df.loc[df[col_sorted].searchsorted(bins[len(bins)-1]):]
	df_counted.loc[len(bins)-1,'Sums']=df_sub[col_tot].sum() #or maybe it's 1?
	return df_counted

# ['created_at', 'user_id', 'user_followers_count', 'retweeted_status_id',
#        'retweeted_status_user_id', 'retweeted_status_retweet_count',
#        'retweeted_status_favorite_count', 'retweeted_status_reply_count',
#        'quoted_status_user_id', 'quoted_status_id', 'Original', 'RT', 'QT',
#        'Reply', 'Media', 'Length']

def type_rates(twt_db,dts):
    # function to report rates of different twitter activity on time series dts
    # this function report original, RT, QT, and replies
    df_mn_counts = pd.DataFrame(index = dts)
    df_alltwt_ord = twt_db.sort_values('created_at')
    df_alltwt_ord = df_alltwt_ord.reset_index(drop=True)

    df_Ori = df_alltwt_ord.loc[df_alltwt_ord['Original']==1,:].reset_index(drop = True)   
    df_counted = ind_set_counts(df_Ori,'created_at',dts)
    df_mn_counts['Original'] = df_counted['Counts'].values

    df_RT = df_alltwt_ord.loc[df_alltwt_ord['RT']==1,:].reset_index(drop = True)  
    df_counted = ind_set_counts(df_RT,'created_at',dts)
    df_mn_counts['RT'] = df_counted['Counts'].values
    
    df_replys = df_alltwt_ord.loc[df_alltwt_ord['Reply']==1,:].reset_index(drop = True)
    df_counted = ind_set_counts(df_replys,'created_at',dts)
    df_mn_counts['Reply'] = df_counted['Counts'].values
    
    df_QT = df_alltwt_ord.loc[df_alltwt_ord['QT']==1,:].reset_index(drop = True)
    df_counted = ind_set_counts(df_QT,'created_at',dts)
    df_mn_counts['Quote'] = df_counted['Counts'].values
    
    return df_mn_counts

def type_rates_all(twt_db,dts):
       # function to report rates of different twitter activity on time series dts
    # this function report all, original, RT, QT, replies, and unique users active
    df_mn_counts = pd.DataFrame(index = dts)
    
    df_alltwt_ord = twt_db.sort_values('created_at')
    df_alltwt_ord = df_alltwt_ord.reset_index(drop=True)

    df_mn_counts = pd.DataFrame(index = dts)
    df_counted = ind_set_counts(df_alltwt_ord,'created_at',dts)
    df_mn_counts['All'] = df_counted['Counts'].values
    
    df_Ori = df_alltwt_ord.loc[df_alltwt_ord['Original']==1,:].reset_index(drop = True)    
    df_counted = ind_set_counts(df_Ori,'created_at',dts)
    df_mn_counts['Original'] = df_counted['Counts'].values
    
    df_Shouts = df_Ori.loc[df_Ori['Media']==0]
    df_Shouts = df_Shouts.loc[df_Shouts['Length']<70].copy()
    df_counted = ind_set_counts(df_Shouts,'created_at',dts)
    df_mn_counts['Shouts'] = df_counted['Counts'].values

    df_RT = df_alltwt_ord.loc[df_alltwt_ord['RT']==1,:].reset_index(drop = True)  
    df_counted = ind_set_counts(df_RT,'created_at',dts)
    df_mn_counts['RT'] = df_counted['Counts'].values
    
    df_replys = df_alltwt_ord.loc[df_alltwt_ord['Reply']==1,:].reset_index(drop = True)
    df_counted = ind_set_counts(df_replys,'created_at',dts)
    df_mn_counts['Reply'] = df_counted['Counts'].values
    
    df_QT = df_alltwt_ord.loc[df_alltwt_ord['QT']==1,:].reset_index(drop = True)
    df_counted = ind_set_counts(df_QT,'created_at',dts)
    df_mn_counts['Quote'] = df_counted['Counts'].values
    
    df_counted = ind_set_unique_counts(df_alltwt_ord,'created_at','user_id',dts)
    df_mn_counts['Users'] = df_counted['Counts'].values
    
    return df_mn_counts

def Streamed_Set_Stats_Accumulated(df_set,ID_Field,ID_value):
    # all the tweets with the same values in the relevant, like all RTs of a tweet specified in 'retweeted_status_id'
    # feature set specific to fields from stream recorded data, not from the academic API that samples surviving DB.
    # df_set is assumed to be chronological.

    feild_set = df_set.loc[df_set[ID_Field]==ID_value,:].reset_index(drop=True)
    cols_counts = []
    cols_acc = []
    # select which feilds to count over based on what is stable in the set
    if len(feild_set['user_id'].unique())<2:
        cols_counts+=['user_followers_count']
    else:
        cols_acc = ['user_followers_count']
        
    if len(feild_set['retweeted_status_id'].unique())<2:
        if not np.isnan(feild_set['retweeted_status_id'].unique()[0]):
            cols_counts+=['retweeted_status_retweet_count', 
                          'retweeted_status_favorite_count',
                          'retweeted_status_reply_count']

    Stats_accumulated = pd.DataFrame(index = feild_set['created_at'])
    for col in cols_counts:
        Stats_accumulated[col] = feild_set[col].values
        Stats_accumulated['GAIN_' + col] = Stats_accumulated[col]-Stats_accumulated[col].iloc[0]
        Stats_accumulated['REL_' + col] = Stats_accumulated['GAIN_' + col]/Stats_accumulated['GAIN_' + col].max() 
        
    if len(cols_acc)>0:
        Stats_accumulated['Exposures'] = feild_set['user_followers_count'].values.cumsum()
        Stats_accumulated['REL_Exposures'] = Stats_accumulated['Exposures']/Stats_accumulated['Exposures'].max()
    return Stats_accumulated


