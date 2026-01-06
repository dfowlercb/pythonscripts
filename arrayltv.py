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

#STATS
merch=sl[2]*sl[4]
cost=sl[3]*sl[4]
date=np.zeros(ol[2].shape[0],dtype=object)
x=np.cumsum(olidx[1])
i=0
for j in enumerate(x):
    y=dct[j[0]][1][:,0]
    date[i:j[1]]=dct[j[0]][1][:,1][np.searchsorted(y,ol[2][i:j[1]])]
    i=j[1]    

years = date.astype('datetime64[Y]').astype(int) + 1970
months = date.astype('datetime64[M]').astype(int) % 12 + 1
days =  date.astype('datetime64')-date.astype('datetime64[M]') + 1

binsort=np.argsort(sl[0])
oid=sl[0][binsort]
unq, bins=np.unique(oid,return_inverse=1)
merchsum=np.bincount(bins,merch[binsort])
costsum=np.bincount(bins,cost[binsort])
idx=np.searchsorted(unq,ol[0])
oidsum=np.array((unq,years[idx],months[idx],days[idx].astype(int),merchsum,merchsum-costsum))
oidsum=oidsum[:, oidsum[3, :].argsort()]
oidsum=oidsum[:, oidsum[2, :].argsort(kind='mergesort')]
oidsum=oidsum[:, oidsum[1, :].argsort(kind='mergesort')]

unq, bins=np.unique(oidsum[1],return_inverse=1)
merchyear=np.bincount(bins,oidsum[4])
profityear=np.bincount(bins,oidsum[5])

from numpy import random

x=random.randint((ol[0].size-1), size=(5))
oidsum[:,x]
ol[:,np.isin(ol[0],oidsum[:,x][0][0])]
sl[:,np.isin(sl[0],oidsum[:,x][0][0])]

#origin and cohort 
olcust=ol[1]
oldate=ol[2]
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

first=np.array((olcust,date,ol[0]))
first=first[:,datesort]
cust=np.unique(first[0],return_index=1)
origin=first[:,cust[1]]
cohort=np.array((origin[0],origin[1].astype('datetime64[Y]').astype(int) + 1970))

#RFM prep - order totals and customer array
binsort=np.argsort(sl[0])
oid=sl[0][binsort]
bins=np.unique(oid,return_inverse=1)[1]
data=sl[2]*sl[4]
binsum=np.bincount(bins,data[binsort])
oidsum=np.array((np.unique(oid),binsum))

olcid=ol [0 : 2]
cid=np.unique(ol[1])

#recency
olcust=ol[1]
oldate=ol[2]
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
olcid=ol [0 : 2]
olcid=olcid [ :, olcid[0].argsort()]
ordersum=oidsum[1][np.searchsorted(olcid[0],oidsum[0])]
olcid=np.vstack((olcid,ordersum))
olcid=olcid [ :, olcid[1].argsort()]
bins=np.unique(olcid[1],return_inverse=1)[1]
monetary=np.bincount(bins,ordersum)


#categories
col=3
x=np.cumsum(slidx[1])
i=0

categories=[]
for j in enumerate(x):
    categories=categories+list(set(dct[j[0]][0][:,col].tolist()))
    i=j[1]    

categories=np.unique(categories)

for i in range(0,len(dct)):
    dct[i][0]=np.delete (dct[i][0],7,axis=1)

for i in range(0,len(dct)):
        dct[i][0]=np.hstack((dct[i][0],np.searchsorted(categories,dct[i][0][:,col]).reshape(-1,1)))

slsku=sl[1].astype(int)
cat=np.zeros(slsku.shape[0],dtype=object)

x=np.cumsum(slidx[1])
i=0

for j in enumerate(x):
    y=dct[j[0]][0][:,0].astype(int)
    cat[i:j[1]]=dct[j[0]][0][:,7][np.searchsorted(y,slsku[i:j[1]])]
    i=j[1]    

merch=sl[2]*sl[4]
cat=np.array([sl[0],cat,merch],dtype=np.float64)
oidcatsort=np.lexsort([cat[1],cat[0]])
cat=cat[:,oidcatsort]
unq,cnt=np.unique(cat[0:2],axis=1,return_counts=1)
bins=np.repeat(np.arange(len(cnt)),cnt)
catmerch=np.bincount(bins,cat[2:].flatten())
catmerch=np.array([unq[0],unq[1],catmerch])
oidrecode=np.unique(unq[0],return_inverse=1)
catmerch=np.vstack((oidrecode[1],catmerch))

from scipy import sparse
row=catmerch[0].astype(int)
col=catmerch[2].astype(int)
data=catmerch[3]
coo = sparse.coo_matrix((data, (row, col)))
sparse.save_npz('/home/dmf/python/npz/coo_bg.npz', coo)

ol=ol[:,np.argsort(ol[0])]
x=np.searchsorted(ol[0],catmerch[1])
catmerch=np.vstack((catmerch,ol[1][x]))
catmerch=catmerch[:,np.argsort(catmerch[4])]
cidrecode=np.unique(catmerch[4],return_inverse=1)
catmerch=np.vstack((catmerch,cidrecode[1]))

#order percentage
coo=coo.tolil()     
j=np.where(categories=='HMS')[0]
hms=np.sum(coo[:,j],axis=1)/np.sum(coo,axis=1)
