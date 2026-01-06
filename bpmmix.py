import numpy as np
from numpy.lib import recfunctions as rfn
from scipy import sparse
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

#order file
arr22=np.load('/home/dfowler/npz/l22.npz')['arr_0']

#add bpm classification to order file
dt=np.dtype([ ('sku','U12'), ('indiv','f8'), ('combo','f8'), ('class','U13') ])
bpm=np.loadtxt('/home/dfowler/csv/bpm.csv',delimiter=',',dtype=dt,skiprows=1)
i=np.where(bpm['class']=='MEDIA')
bpm['class'][i]='OTHER'
i=np.where(bpm['class']=='')
bpm['class'][i]='OTHER'
bpm=np.sort(bpm,order='sku')
i=np.searchsorted(bpm['sku'],arr22['SKU'])
classification=bpm['class'][i]
arr22=rfn.append_fields(arr22,'class',classification,usemask=False)

#need weight file using random numbers in the meantime
dt=np.dtype( [('sku','U13'),('wtper10','f8')] )
wt=np.loadtxt('/home/dfowler/csv/skuweight.csv',delimiter=',',converters = {1: lambda s: float(s or 0)},dtype=dt,skiprows=1,encoding='latin1')
wt=np.sort(wt,order='sku')
idx=np.searchsorted(wt['sku'],arr22['SKU'])
weight=wt['wtper10'][idx]
weight=weight/10
arr22=rfn.append_fields(arr22,'wt',weight,usemask=False)

#lowcost boolean
lowcost=np.repeat(0,arr22.shape[0])
lowcost[arr22['class']=='BPM']=1
lowcost[np.logical_and(arr22['class']=='OTHER',arr22['UNIT_COST']<11.70)]=1
arr22=rfn.append_fields(arr22,'lowcost',lowcost,usemask=False)

#prepare for matrix
oid=np.unique(arr22['ORDER_NO'],return_inverse=True)
classify=np.unique(arr22['class'],return_inverse=True)
lowcost=np.unique(arr22['lowcost'],return_inverse=True)

#sparse matrices
 #weight
row=oid[1]
col=classify[1]
shape=tuple((oid[0].size,classify[0].size))
data=arr22['wt']
csr=sparse.coo_matrix((data,(row,col)),shape=shape).tocsr()
print(csr.shape)
weight=csr.toarray()
 #low cost
row=oid[1]
col=arr22['lowcost']
shape=tuple((oid[0].size,lowcost[0].size))
data=arr22['UNIT_COST']
csr=sparse.coo_matrix((data,(row,col)),shape=shape).tocsr()
print(csr.shape)
lc=csr.toarray()

#structured array
x=np.hstack((weight,lc))
dt=np.dtype([ ('oid','i8')])
oid=np.reshape(oid[0],(oid[0].size,1))
arr0=rfn.unstructured_to_structured(oid,dtype=dt)
arr0=rfn.append_fields(arr0,names=['w','x','y','z'],data=(x[:,0],x[:,1],x[:,2],x[:,3]),dtypes=('f8','f8','f8','f8'),usemask=False)


