# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:52:01 2022

@author: quru
"""

import numpy as np
import pandas as pd
import random
from math import isnan

use_sample = False
j = 2000
def random_excel(df,j):
    df = df.dropna(axis = 1, how = 'all')
    l = len(df)
    dfid = [i for i in df.columns if 'id' in i and df[i].dtypes == 'int64']
    df[dfid] = df[dfid].astype('object')
    df_obj = df.select_dtypes(include=['object'])
    df_float = df.select_dtypes(include=['float64'])
    df_other = df.select_dtypes(exclude=['object','datetime64','float64'])
    df_time = df.select_dtypes(include=['datetime64'])
    #df_obj
    if len(df_obj.columns)!=0: 
        a = {}
        for i in df_obj:
            a[i] = df_obj.loc[:,i].unique()
            a[i] = [x for x in a[i] if str(x) != 'nan' and str(x)!='?']
     
        c = []
        for k in a:
            a1 = random.choices(a[k],k = l)
            c.append(a1) 
    
        out_obj = pd.concat([pd.Series(x) for x in c], axis=1)
        out_obj.columns = df_obj.columns
    else:
        out_obj = []
    
    #df_other
    if len(df_other.columns) !=0:
        b = {}
        for i in df_other:
            b[i]=[min(df_other.loc[:,i]), max(df_other.loc[:,i])]
    
    
        d = []
        for k in b:
            b1 = np.random.randint(low=b[k][0], high=b[k][1]+1, size=l)
            d.append(b1)
    
        out_other = pd.concat([pd.Series(x) for x in d], axis=1)
        out_other.columns = df_other.columns
    else:
        out_other = []
   
    if len(df_float.columns)!=0:
        e = {}
        for i in df_float:
            e[i] = [np.mean(df_float.loc[:,i][df_float.loc[:,i]!=np.inf]), np.std(df_float.loc[:,i][df_float.loc[:,i]!=np.inf])]
        
        f = []
        for k in e:
            e1 = np.random.normal(e[k][0], e[k][1], size=l)
            f.append(e1)
            
        out_flt = pd.concat([pd.Series(x) for x in f], axis=1)
        out_flt.columns = df_float.columns
    else:
        out_flt = []
#we have 4 types : out_obj, out_other, out_flt, time        
        
    if len(df_obj.columns) == 0:
        out = pd.concat([out_flt, df_time, out_other], axis = 1)
        cols = list(df.columns)
        out = out[cols]
    elif len(df_other.columns) == 0:
        out = pd.concat([out_obj, df_time, out_flt], axis = 1)
        cols = list(df.columns)
        out = out[cols]
    elif len(df_float.columns) == 0:
        out = pd.concat([out_obj, df_time,out_other], axis = 1)
        cols = list(df.columns)
        out = out[cols]
    else:
        out = pd.concat([out_obj,out_flt, out_other, df_time] ,axis = 1)  
        cols = list(df.columns)
        out = out[cols]
    if use_sample:
        out = out.sample(j)    
    return out



promo_cld = pd.read_pickle('promoCalendarDF.pkl')
bc = pd.read_pickle('newView1.pkl')
hs = pd.read_pickle('hs.pkl')
obj_l = ['Model Version','Version', 'Contingency','Updated Promo'] 
for i in obj_l:
    bc[i] = bc[i].astype('object')
bc['Promo Cost Per Incremental Net Add'] == np.inf
np.std(bc['Promo Cost Per Incremental Net Add'][bc['Promo Cost Per Incremental Net Add']!=np.inf])
bc.dtypes

random_cld = random_excel(promo_cld)
random_bc = random_excel(bc,j=2000)
random_hs = random_excel(hs, j)



random_cld.to_csv('random/random_Promo_Cal.csv',index=False)
random_bc.to_csv('random/random_business_case_v.csv',index=False)
random_bc_2k.to_csv('random/random_business_case_v_2k.csv',index=False)
random_hs_1.to_csv('random/random_hs.csv',index=False)


# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:57:41 2021

@author: quru
"""

import numpy as np
import pandas as pd
import random
from math import isnan


promo = pd.read_excel('Promo.xlsx', header = 0)
traf = pd.read_excel('Traffic.xlsx', header = 0)
comlink = pd.read_excel('comlink_promo.xlsx', header = 0)
media = pd.read_excel('media_spend.xlsx', header = 0)
Promo_ML = pd.read_excel('Promo_ML.xlsx', header = 0)
Promo_ga = pd.read_excel('grossadds_promo.xlsx', header = 0 )
df = promo.copy()
df = df.dropna(axis = 1, how = 'all')
df.info()
a = [i for i in promo.columns if 'id' in i and promo[i].dtypes == 'int64']

df_obj = df.select_dtypes(include=['object'])
aid = promo[a].astype('object')

a 
aid.select_dtypes(include=['int64'])


def random_excel(df):
    df = df.dropna(axis = 1, how = 'all')
    l = len(df)
    dfid = [i for i in df.columns if 'id' in i and df[i].dtypes == 'int64']
    df[dfid] = df[dfid].astype('object')
    df_obj = df.select_dtypes(include=['object'])
    df_other = df.select_dtypes(exclude=['object','datetime64'])
    df_time = df.select_dtypes(include=['datetime64'])
    #df_obj
    if len(df_obj.columns)!=0: 
        a = {}
        for i in df_obj:
            a[i] = df_obj.loc[:,i].unique()
            a[i] = [x for x in a[i] if str(x) != 'nan' and str(x)!='?']
     
        c = []
        for k in a:
            a1 = random.choices(a[k],k = l)
            c.append(a1) 
    
        out_obj = pd.concat([pd.Series(x) for x in c], axis=1)
        out_obj.columns = df_obj.columns
    else:
        out_obj = []
    
    #df_other
    if len(df_other.columns) !=0:
        b = {}
        for i in df_other:
            b[i]=[min(df_other.loc[:,i]), max(df_other.loc[:,i])]
    
    
        d = []
        for k in b:
            b1 = np.random.randint(low=b[k][0], high=b[k][1]+1, size=(l))
            d.append(b1)
    
        out_flt = pd.concat([pd.Series(x) for x in d], axis=1)
        out_flt.columns = df_other.columns
    else:
        out_flt = []
        
    if len(df_obj.columns) == 0:
        out = pd.concat([out_flt, df_time], axis = 1)
        cols = list(df.columns)
        out = out[cols]
    elif len(df_other.columns) == 0:
        out = pd.concat([out_obj, df_time], axis = 1)
        cols = list(df.columns)
        out = out[cols]
    else:
        out1 = pd.concat([out_obj,out_flt] ,axis = 1)
        out = pd.concat([out1, df_time], axis = 1)
        cols = list(df.columns)
        out = out[cols]
    return out
        
    
    
random_promo = random_excel(promo) 
random_traffic = random_excel(traf) 
random_comlink = random_excel(comlink)
random_media = random_excel(media)  
random_PromoML = random_excel(Promo_ML) 
random_Promoga = random_excel(Promo_ga) 
 
random_promo.to_csv('random/random_Promo.csv',index=False)
random_traffic.to_csv('random/random_traffic.csv',index=False)
random_comlink.to_csv('random/random_comlink.csv',index=False)
random_media.to_csv('random/random_media.csv',index=False)
random_PromoML.to_csv('random/random_PromoML.csv',index=False)
random_Promoga.to_csv('random/random_Promoga.csv',index=False)


sqlstr = 'select * from NTL_PRD_QMTBLS.YIN_CONSUMER_GROSSADDS_REGION where datetime >= 1190101 '
grossadds = readTeradata(host,username,password,sqlstr)

random_ga = random_excel(grossadds)

random_ga.to_csv('random/random_grossadds.csv',index=False)




