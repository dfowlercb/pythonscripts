import numpy as np
from scipy import sparse
import os
from numpy import genfromtxt
from sklearn.preprocessing import scale
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

def load(monthlevel,*features):
    import os
    os.chdir('/home/dmf/python/npz')
    fname=''.join([monthlevel,'hilo.npz'])
    arrays=np.load(fname,allow_pickle=True)
    for x in np.arange(len(features)):
        globals()[features[x]]=arrays[features[x]]

load('olfeb','cid','oidol','date')
load('llfeb','sku','oid','price','units','parts')

#LOAD FILES
 #mainframe customer files
os.chdir('/home/dmf/R/projects/consumer/mayjun22')
mailed=genfromtxt('dydids.csv',skip_header=1,dtype=[('CID','int64')])
orders=genfromtxt('duzmldflowkeycustno.csv',delimiter=',',dtype=[('CID','int64'),('OID','int64')],skip_header=1)
history=genfromtxt('dydmldords3yr.csv',delimiter=',',dtype=[('CID','int64'),('OID','int64')],skip_header=1)

#restrict to scope of mainframe files and remove category ADJ
def scope(idx):
    ol_tup=('cid','oidol','date')
    ll_tup=('sku','oid','price','units')
    for x in ol_tup:
        globals()[x]=globals()[x][idx]
    
    i=np.isin(oid,oidol)
    for x in ll_tup:
        globals()[x]=globals()[x][i]

scope(np.isin(oidol,history['OID']))
scope(np.isin(cid,mailed['CID']))

adj=np.where(parts[:,4]=='ADJ')
i=np.isin( sku, adj, invert = True)
for x in ('sku','oid','price','units'):
    vars()[x]=vars()[x][i]

i=np.isin(oidol,np.unique(oid))
for x in ('cid','oidol','date'):
    vars()[x]=vars()[x][i]

#sort order level
i=np.argsort(oidol)
for x in ('cid','oidol','date'):
    vars()[x]=vars()[x][i]

ll_link=np.searchsorted(oidol,oid)

#exclusions
cidnum=np.unique(cid[ll_link],return_inverse=1)
I=cidnum[1]
J=parts[sku,7].astype(int)
V=units*price
shape=tuple((cidnum[0].size,np.unique(parts[:,7]).size))
bg=sparse.coo_matrix((V,(I,J)),shape=shape).tocsr()

J=parts[sku,8].astype(int)
shape=tuple((cidnum[0].size,np.unique(parts[:,8]).size))
t1=sparse.coo_matrix((V,(I,J)),shape=shape).tocsr()

J=parts[sku,9].astype(int)
shape=tuple((cidnum[0].size,np.unique(parts[:,9]).size))
t2=sparse.coo_matrix((V,(I,J)),shape=shape).tocsr()

J=parts[sku,10].astype(int)
shape=tuple((cidnum[0].size,np.unique(parts[:,10]).size))
t3=sparse.coo_matrix((V,(I,J)),shape=shape).tocsr()

bgkey=np.unique(np.array((parts[:,2],parts[:,7])),axis=1).T
bgselect=bgkey[np.isin(bgkey[:,0],['ACADEMIC','CHURCH INTEREST','HOMESCHOOLING']),1].astype(int)
t1key=np.unique(np.array((parts[:,3],parts[:,8])),axis=1).T
t1select=t1key[np.isin(t1key[:,0],['10']),1].astype(int)
t2key=np.unique(np.array((parts[:,4],parts[:,9])),axis=1).T
t2select=t2key[np.isin(t2key[:,0],['ACC','CLG','VBS']),1].astype(int)
t3key=np.unique(np.array((parts[:,5],parts[:,10])),axis=1).T
t3select=t3key[np.isin(t3key[:,0],['VB']),1].astype(int)

keep=np.where((bg[:,bgselect[0]]/np.sum(bg,axis=1)<.1) & (bg[:,bgselect[1]]/np.sum(bg,axis=1)<.1) & (bg[:,bgselect[2]]/np.sum(bg,axis=1)<.1) & (t1[:,t1select[0]]/np.sum(t1,axis=1)<.1) & (t2[:,t2select[0]]/np.sum(t2,axis=1)<.1) & (t2[:,t2select[1]]/np.sum(t2,axis=1)<.1) & (t2[:,t2select[2]]/np.sum(t2,axis=1)<.1) & (t3[:,t3select[0]]/np.sum(t3,axis=1)<.1) )[0]

scope(np.isin(cid,cidnum[0][keep]))

ll_link=np.searchsorted(oidol,oid)
cidnum=np.unique(cid[ll_link],return_inverse=1)

#%%category ordered yesno matrix
yesnokey=np.unique(parts[:,1])
I=cidnum[1]
J=parts[sku,6].astype(int)
V=units*price
shape=tuple((cidnum[0].size,yesnokey.size))
cat=sparse.coo_matrix((V,(I,J)),shape=shape).toarray()
cat[cat!=0]=1
i=np.isin(cidnum[0],orders['CID'])
orderedcount=np.where(i==1)[0].size
ordered=np.sum(cat[i],axis=0)
i=np.isin(cidnum[0],orders['CID'],invert=1)
notorderedcount=np.where(i==1)[0].size
notordered=np.sum(cat[i],axis=0)
i=np.where(ordered+notordered>=100)
catmat=np.array((notordered,ordered,notordered+ordered,notordered/notorderedcount,ordered/orderedcount,(ordered/orderedcount)/(notordered/notorderedcount)))

##NEED TO REDUCE ABOVE TO TOTAL>=100
ma_catmatscore=catmat[5,:]
ma_catmatscore[catmat[2,:]<100]=np.nan
ma_catmatscore[9330]=np.nan
ma_catmatscore=np.ma.array(ma_catmatscore,mask=np.isnan(ma_catmatscore))

#%% calculate percentile scores
percentiles=[]
for i in np.arange(101):
    percentiles.append(np.percentile(ma_catmatscore[ma_catmatscore.mask==False] , i))

percentiles = np.asarray(percentiles)
percentilescore = np.searchsorted(percentiles, ma_catmatscore)

i=np.searchsorted(yesnokey,parts[:,1])
parts=np.hstack((parts,percentilescore[i].reshape(-1,1)))
select=np.array((cidnum[1],units,price,parts[sku,6].astype(int),parts[sku,11].astype('float')))
i=np.unique(parts[np.isin(parts[:,1],yesnokey),6])
select=select[:,np.isin(select[3],i.astype(int))]

select=select[:,select[3]!=9330]
#%% weighted average matrix
rows=select[0].astype(int)
cols=np.repeat(0,rows.size)
data=select[1]*select[4]
shape=tuple((cidnum[0].size ,1 ))
weight_score_sum=sparse.coo_matrix((data,(rows,cols)),shape=shape)
data=select[1]
weight_sum=sparse.coo_matrix((data,(rows,cols)),shape=shape)
weighted_average=weight_score_sum/weight_sum

#NEED TO ONLY ACCEPT PERCENTILE <50% PER R MODEL
idx = np.where(weighted_average<=50)[0]
keep=cidnum[0][idx]
scope(np.isin(cid,keep))

ll_link = np.searchsorted(oidol,oid)
cidnum = np.unique ( cid, return_inverse = True )

#RFM + unique skus and unique markets
 #recency
daysago=(np.datetime64('today','D')-date.astype('datetime64[D]')).astype(int)
datesort=np.lexsort((daysago,cid))

rec=np.array((cid,daysago))
rec=rec[:,datesort]
cust=np.unique(rec[0],return_index=1)
recency=rec[:,cust[1]]

 #frequency
freq = np.unique(cid,return_counts=1)
frequency = np.array( [ freq[0], freq[1] ] )

#monetary
rows=cidnum[1][ll_link]
cols=np.repeat(0,rows.size)
data=units*price
monetary=sparse.coo_matrix((data,(rows,cols))).toarray()

#unique skus
skucount=np.unique(np.array([cid[ll_link],sku]),axis=1)
uniquesku=np.unique(skucount[0,:],return_counts=1)

#unique markets
marketcount=np.unique(np.array([cid[ll_link],parts[sku,9].astype(int)]),axis=1)
uniquemarket=np.unique(marketcount[0,:],return_counts=1)

score=((scale(-recency[1]))+scale(freq[1])+scale(monetary).T.flatten()+scale(uniquesku[1])+scale(uniquemarket[1]))/5
score=np.round(score*10000).astype(int)
i=np.where(monetary<1000)[0]
export=np.array( [recency[0][i], score.flatten()[i] ] )
export=export[:,(-export[1]).argsort()]
export=export[:,:400000]

percentiles=[]
for i in np.arange(10,101,10):
    percentiles.append(np.percentile(export[1] , i))

score=np.searchsorted(percentiles,export[1])+1
export[1]=score
np.savetxt('/home/dmf/Documents/mayjun22coreselect.txt',export.T,delimiter='\t',fmt='%i',header=('CID\tSCORE'),comments='')

counts=np.unique(export[1],return_counts=True)
counts=np.array((counts[0],counts[1])).T
np.savetxt('/home/dmf/Documents/mayjun22scorecounts.txt',counts, fmt='%i')

#INTEGRITY CHECK WITH R VERSION
x=np.genfromtxt('/home/dmf/R/projects/consumer/mayjun22/mayjun22coreselect.txt',dtype='int64')
y=np.intersect1d(export[0],x[:,0])
print(y.size)

