import numpy as np
from numpy.lib import recfunctions as rfn
np.set_printoptions(suppress=True)

x=np.load('/home/dfowler/npz/l2019_2024a.npz')['arr_0']
dt=np.dtype([('SKU','U21'),('CAT','U9')])

parts=np.loadtxt('/home/dfowler/csv/parts.txt',delimiter=',',dtype=dt,usecols=(6,11),encoding='latin1',skiprows=1)
parts=np.sort(parts,order='SKU')
idx=np.searchsorted(parts['SKU'],x['SKU'])
x=rfn.append_fields(x,'CAT',parts['CAT'][idx],usemask=False)

#idx=np.logical_or(np.char.startswith(x['CAT'],'05-'),np.char.startswith(x['CAT'],'01-'))
idx=x['SKU']=='260039'
cid=np.unique(x[idx]['CM_ID'])
y=x[np.isin(x['CM_ID'],cid)]

customercount=np.unique(y[['SKU','CM_ID']],return_counts=True)
sku=np.unique(customercount[0]['SKU'],return_counts=True)
export=rfn.unstructured_to_structured(sku[0].reshape(-1,1))
export=rfn.append_fields(export,'CUSTCOUNT',sku[1],usemask=False)
export=export[np.flip(np.argsort(export,order='CUSTCOUNT'))]
np.savetxt('/home/dfowler/documents/x.txt',export,fmt='%s')

idx=x['CAT']=='07-DRA-BL'
drabl=x[idx]['CM_ID']
y=x[np.isin(x['CM_ID'],drabl)]

export=np.unique(y['CAT'],return_counts=True)
export=np.stack((export[0],export[1])).T
np.savetxt('/home/dfowler/documents/y.txt',export,fmt='%s')
