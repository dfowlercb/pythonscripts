import numpy as np
from datetime import datetime

def convertdatebib(x):
    x=np.datetime64(datetime.strptime(x,'%m/%d/%y').date())
    return x

def partstoint(x):
    i=np.searchsorted(parts,x)
    return(i)

parts=np.genfromtxt('/home/dmf/csv/parts.txt',delimiter=',',skip_header=1,encoding='latin1',usecols=6,dtype='U16',converters={6: lambda s: str(s or '')})
parts=np.sort(parts)

dt=np.dtype([ ('oid','i4'),('date','M8[D]'), ('sku','i4'), ('price','f4' ), ('units','i4' ) ])
bib=np.genfromtxt('/home/dmf/csv/bib.txt',usecols=(1,2,4,6,7),skip_header=1,dtype=dt,delimiter='\t',converters={2: convertdatebib,4:partstoint},encoding='latin1')
idxbib=np.where(bib['date']>=np.datetime64('2022-01-01'))[0]
bib=bib[idxbib]
np.min(bib['date'])
np.max(bib['date'])

dt=np.dtype([ ('oid','i4'), ('date','i4'),('sku','i4'), ('price','f4' ), ('units','i4' ) ])
high=np.genfromtxt('/home/dmf/csv/high.txt',usecols=(0,2,6,8,9),skip_header=1,dtype=dt,delimiter='\t',encoding='latin1',converters={6:partstoint})
idxhigh=np.where(high['date']>=np.datetime64('2022-01-01').astype(int))[0]
high=high[idxhigh]
np.min(high['date'].astype('datetime64[D]'))
np.max(high['date'].astype('datetime64[D]'))

#EXCLUDE RESHIPS
reships=np.unique(high[high['sku']==np.where(parts=='RESHIP')[0]]['oid'])
high=high[np.isin(high['oid'],reships,invert=True)]

#COUNTS
np.unique(high['oid']).size
np.unique(bib['oid']).size
np.unique(high['oid']).size+np.unique(bib['oid']).size

np.nansum(high['units']*high['price'])
np.sum(bib['units']*bib['price'])
np.nansum(high['units']*high['price'])+np.sum(bib['units']*bib['price'])
