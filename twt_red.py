# functions for analysis of tweets from reduced anonymised sets

import sys
import os
import time
import datetime as dt
import math
import numpy as np 
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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


def tweet_rates(twt_set,sample_rate,smoothing_rate):
    time_start = twt_set['created_at'].min()
    time_end = twt_set['created_at'].max()
    print([time_start,time_end])
    
    timestamps_ind = pd.date_range(time_start,time_end, freq=sample_rate)
    minute_smoothing=int(pd.Timedelta(smoothing_rate)/pd.Timedelta(sample_rate))
        
    df_Ori = twt_set.loc[twt_set['Original']==1,:].reset_index()   
    df_RT =twt_set.loc[twt_set['RT']==1,:].reset_index()  
    df_replys = twt_set.loc[twt_set['Reply']==1,:].reset_index()
    df_QT = twt_set.loc[twt_set['QT']==1,:].reset_index()
    df_inter = pd.concat([df_replys,df_QT]).sort_values('created_at').reset_index(drop=True)
    df_Shouts = df_Ori.loc[df_Ori['Media']==0,:]
    df_Shouts = df_Shouts.loc[df_Shouts['Length']<60,:].copy().reset_index()

    count_field = 'retweeted_status_favorite_count'
    top_number = 1000
    ID_Field = 'retweeted_status_id'
    Feild_sorted = twt_set[ID_Field].value_counts()
    

    df_TS = pd.DataFrame(index = timestamps_ind)

    for i in range(top_number):
        ID_value = Feild_sorted.index[i]
        Stats_accumulated = Streamed_Set_Stats_Accumulated(twt_set,ID_Field,ID_value)
        A = Stats_accumulated[~Stats_accumulated.index.duplicated(keep='last')]
        df_reindexed = A.reindex(timestamps_ind) 
        df_reindexed.interpolate(method='linear',inplace = True)
        df_TS[i] = df_reindexed[count_field]

    df_S_Acc = pd.DataFrame(index = timestamps_ind)
    df_S_Acc['Favorites rate*'] = df_TS.diff().rolling(minute_smoothing).sum().sum(axis = 1)
    df_counted = ind_set_counts(df_Ori,'created_at',timestamps_ind)
    df_S_Acc['Original Tweet rate'] = df_counted['Counts'].rolling(minute_smoothing).sum().values
    df_counted = ind_set_counts(df_RT,'created_at',timestamps_ind)
    df_S_Acc['Retweets rate'] = df_counted['Counts'].rolling(minute_smoothing).sum().values
    df_counted = ind_set_counts(twt_set,'created_at',timestamps_ind)
    df_S_Acc['All Tweet rate'] = df_counted['Counts'].rolling(minute_smoothing).sum().values
    df_counted = ind_set_counts(df_inter,'created_at',timestamps_ind)
    df_S_Acc['Interaction rate'] = df_counted['Counts'].rolling(minute_smoothing).sum().values
    

    df_counted = ind_set_counts(df_Shouts,'created_at',timestamps_ind)
    df_S_Acc['Shouts rate'] = df_counted['Counts'].rolling(minute_smoothing).sum().values
    
    for c in df_S_Acc.columns:
        df_S_Acc['Scaled ' + c] = df_S_Acc[c]/df_S_Acc[c].median()

    return df_S_Acc

def cevents(cdets):
    concert_times=pd.read_csv(cdets['data_loc'] + cdets['event_file'])
    concert_times["starttime"] = pd.to_datetime(concert_times["starttime"])+pd.Timedelta(cdets['event_offset']) # added 6 minutes, so songs begin less than a minute before the first yell tweet of recognition
    concert_times=concert_times.set_index("starttime", drop=True)
    return concert_times

def concert_phases(ax,yrange,ctimes):
    ax.set_xlabel('Time (UTC)')
    ax.set_xticks(ctimes.index)
    ax.set_xticklabels(ctimes['event'],rotation=50,ha='right')
    
    ax.set_ylim(yrange)
    ax.grid()
    ax.margins(0)

    for i in range(len(ctimes)-1):
        r= ctimes.iloc[i,:]
        ei = r.name
        ej = ctimes.index[i+1]
        if r['event_type'].endswith('Music'):
            ax.axvspan(ei, ej,yrange[0],yrange[1], facecolor='blue', alpha=0.1)
        if r['event_type'].endswith('Talk'):
            ax.axvspan(ei, ej,yrange[0],yrange[1], facecolor='orange', alpha=0.1)
        if r['event_type'].startswith('VCR'):
            ax.axvspan(ei, ej,yrange[0],yrange[1], facecolor='green', alpha=0.1)
        if r['event_type'].startswith('Not'):
            ax.axvspan(ei, ej,yrange[0],yrange[1], facecolor='red', alpha=0.1)
    return

def segment_rates(segments,twt_set,tweet_rates,):
    df_RT = twt_set.loc[twt_set['retweeted_status_id'].notna(),:].reset_index()

    df_Ori = twt_set.loc[twt_set['retweeted_status_id'].isna(),:]
    df_Ori = df_Ori.loc[df_Ori['quoted_status_id'].isna(),:] 
    df_Ori = df_Ori.loc[df_Ori['in_reply_to_status_id'].isna(),:].reset_index()
    
    print([len(twt_set),len(df_RT),len(df_Ori)])
    
    concert_t = segments.copy()
    
    seg_dur = []
    ori_count = []
    ori_rate = []
    ori_rate_ch = []
    rt_count = []
    rt_rate = []
    rt_rate_ch = []
    al_count = []
    al_rate = []
    al_rate_ch = []
    for event_n in range(len(concert_t)-1):
        
        ev_time = concert_t.index[event_n]
        dur = (concert_t.index[event_n+1].tz_localize(None)-concert_t.index[event_n].tz_localize(None))
        seg_dur.append(dur.seconds)
        
        a = ind_set(df_Ori,'created_at',concert_t.index[event_n],concert_t.index[event_n+1])
        ori_count.append(len(a))
        ori_rate.append(60*len(a)/dur.seconds)
        
        a = ind_set(df_RT,'created_at',concert_t.index[event_n],concert_t.index[event_n+1])
        rt_count.append(len(a))
        rt_rate.append(60*len(a)/dur.seconds)
        
        a = ind_set(twt_set,'created_at',concert_t.index[event_n],concert_t.index[event_n+1])
        al_count.append(len(a))
        al_rate.append(60*len(a)/dur.seconds)
        
        ratediffs= tweet_rates.loc[ev_time:ev_time + pd.Timedelta(minutes=1)].mean()/tweet_rates.loc[ev_time - pd.Timedelta(minutes=1):ev_time].mean()
        ori_rate_ch.append(ratediffs['Original Tweet rate'])
        rt_rate_ch.append(ratediffs['Retweets rate'])
        al_rate_ch.append(ratediffs['All Tweet rate'])

    dur = concert_t.index[-1].tz_localize(None)-concert_t.index[0].tz_localize(None)
    seg_dur.append(dur)
    a = ind_set(df_Ori,'created_at',concert_t.index[0],concert_t.index[-1])
    ori_count.append(len(a))
    ori_rate.append(60*len(a)/dur.seconds)
    a = ind_set(df_RT,'created_at',concert_t.index[0],concert_t.index[-1])
    rt_count.append(len(a))
    rt_rate.append(60*len(a)/dur.seconds)
    a = ind_set(twt_set,'created_at',concert_t.index[0],concert_t.index[-1])
    al_count.append(len(a))
    al_rate.append(60*len(a)/dur.seconds)
    ori_rate_ch.append(-1)
    rt_rate_ch.append(-1)
    al_rate_ch.append(-1)

    concert_t.loc[:,'Segment Durations'] = seg_dur
    concert_t.loc[:,'Original Tweets'] = ori_count
    concert_t.loc[:,'Original Tweet Rate'] = ori_rate
    ori_rate = np.array(ori_rate)
    concert_t.loc[:,'Original Rate Shift'] =  np.append(np.array([1.0]),ori_rate[1:]/ori_rate[:-1])
    concert_t.loc[:,'Original Rate Cusp'] = ori_rate_ch

    concert_t.loc[:,'Retweet Tweets'] = rt_count
    concert_t.loc[:,'Retweet Tweet Rate'] = rt_rate
    rt_rate = np.array(rt_rate)
    concert_t.loc[:,'Retweet Rate Shift'] = np.append(np.array([1.0]),rt_rate[1:]/rt_rate[:-1])
    concert_t.loc[:,'Retweet Rate Cusp'] = rt_rate_ch
    
    concert_t.loc[:,'All Tweets'] = al_count
    concert_t.loc[:,'All Tweet Rate'] = al_rate
    al_rate = np.array(al_rate)
    concert_t.loc[:,'All Tweet Rate Shift'] = np.append(np.array([1.0]),al_rate[1:]/al_rate[:-1])
    concert_t.loc[:,'All Tweet Rate Cusp'] = al_rate_ch
    

    return concert_t.iloc[1:len(concert_t)-1,:]

# https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph
def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)