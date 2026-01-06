import numpy as np
import gc
import os
from numpy import char
import pickle

os.chdir('/home/dmf/python/npz')
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

arrays=np.load('ordersall.npz',allow_pickle=1)
ol=arrays['ol']
sl=arrays['sl']
olidx=arrays['olidx']
slidx=arrays['slidx']

with open("dct.txt", "rb") as fp:   
    dct = pickle.load(fp)

import numpy.lib.recfunctions as rf
dt=np.dtype([('oid','i8'),('cid','i8'),('dates','i8'),('channel','i8'),('zipcode','i8'),('new','i8'),('pay','i8'),('hid','i8')])
ol1=rf.unstructured_to_structured(ol.T,dtype=dt)
dt=np.dtype([('oid','f8'),('sku','f8'),('price','f8'),('cost','f8'),('units','f8'),('pah','f8'),('tax','f8'),('status','f8')])
sl1=rf.unstructured_to_structured(sl.T,dtype=dt)

#STATS
merch=sl1['price']*sl1['units']
cost=sl1['cost']*sl1['units']
date=np.zeros(ol1['dates'].shape[0],dtype=object)
x=np.cumsum(olidx[1])
i=0
for j in enumerate(x):
    y=dct[j[0]][1][:,0]
    date[i:j[1]]=dct[j[0]][1][:,1][np.searchsorted(y,ol1['dates'][i:j[1]])]
    i=j[1]    

years = date.astype('datetime64[Y]').astype(int) + 1970
months = date.astype('datetime64[M]').astype(int) % 12 + 1
days =  date.astype('datetime64')-date.astype('datetime64[M]') + 1

binsort=np.argsort(sl1['oid'])
oid=sl1['oid'][binsort]
unq, bins=np.unique(oid,return_inverse=1)
merchsum=np.bincount(bins,merch[binsort])
costsum=np.bincount(bins,cost[binsort])
idx=np.searchsorted(unq,ol1['oid'])
oidsum=np.array((unq,years[idx],months[idx],days[idx].astype(int),merchsum,merchsum-costsum))
oidsum=oidsum[:, oidsum[3, :].argsort()]
oidsum=oidsum[:, oidsum[2, :].argsort(kind='mergesort')]
oidsum=oidsum[:, oidsum[1, :].argsort(kind='mergesort')]

unq, bins=np.unique(oidsum[1],return_inverse=1)
merchyear=np.bincount(bins,oidsum[4])
profityear=np.bincount(bins,oidsum[5])

from numpy import random

x=random.randint((ol1['oid'].size-1), size=(5))
oidsum[:,x]
ol[:,np.isin(ol1['oid'],oidsum[:,x][0][0])]
sl[:,np.isin(sl1['oid'],oidsum[:,x][0][0])]

#origin and cohort 
olcust=ol1['cid']
oldate=ol1['dates']
date=np.zeros(oldate.shape[0],dtype=object)
x=np.cumsum(olidx[1])
i=0
for j in enumerate(x):
    y=dct[j[0]][1][:,0]
    date[i:j[1]]=dct[j[0]][1][:,1][np.searchsorted(y,oldate[i:j[1]])]
    i=j[1]    

years = date.astype('datetime64[Y]').astype(int) + 1970
months = date.astype('datetime64[M]').astype(int) % 12 + 1
days =  date.astype('datetime64')-date.astype('datetime64[M]') + 1
datesort=np.lexsort((days,months,years,olcust))

first=np.array((olcust,date,ol1['oid']))
first=first[:,datesort]
cust=np.unique(first[0],return_index=1)
origin=first[:,cust[1]]
cohort=np.array((origin[0],origin[1].astype('datetime64[Y]').astype(int) + 1970))

#RFM prep - order totals and customer array
binsort=np.argsort(sl1['oid'])
oid=sl1['oid'][binsort]
bins=np.unique(oid,return_inverse=1)[1]
data=sl1['price']*sl1['units']
binsum=np.bincount(bins,data[binsort])
oidsum=np.array((np.unique(oid),binsum))

olcid=np.array([ol1['oid'],ol1['cid']])
cid=np.unique(ol[1])

#recency
olcust=ol1['cid']
oldate=ol1['dates']
date=np.zeros(oldate.shape[0],dtype=object)
x=np.cumsum(olidx[1])
i=0
for j in enumerate(x):
    y=dct[j[0]][1][:,0]
    date[i:j[1]]=dct[j[0]][1][:,1][np.searchsorted(y,oldate[i:j[1]])]
    i=j[1]    

daysago=np.datetime64('today').astype('datetime64')-date.astype('datetime64')
years = date.astype('datetime64[Y]').astype(int) + 1970
months = date.astype('datetime64[M]').astype(int) % 12 + 1
days =  date.astype('datetime64')-date.astype('datetime64[M]') + 1
datesort=np.lexsort((-days,-months,-years,olcust))

rec=np.array((olcust,date,daysago))
rec=rec[:,datesort]
cust=np.unique(rec[0],return_index=1)
recency=rec[:,cust[1]]

#frequency
olcid=olcid [ :, olcid[1].argsort()]
frequency=np.unique(olcid[1],return_counts=1)[1]

#monetary
olcid=np.array([ol1['oid'],ol1['cid']])
olcid=olcid [ :, olcid[0].argsort()]
ordersum=oidsum[1][np.searchsorted(olcid[0],oidsum[0])]
olcid=np.vstack((olcid,ordersum))
olcid=olcid [ :, olcid[1].argsort()]
bins=np.unique(olcid[1],return_inverse=1)[1]
monetary=np.bincount(bins,ordersum)

#T2
dct2=dct[0][0][:,(0,4)]
np.unique(dct2,axis=0)
slsku=sl[1]
t2=np.zeros(slsku.shape[0],dtype=object)
x=np.cumsum(slidx[1])
i=0
for j in enumerate(x):
    y=dct[j[0]][0][:,0].astype(int)
    t2[i:j[1]]=dct[j[0]][0][:,5][np.searchsorted(y,slsku[i:j[1]])]
    i=j[1]    


