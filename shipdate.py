import numpy as np
from numpy import genfromtxt
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

arrays=np.load('/home/dmf/python/npz/lldechilo.npz',allow_pickle=1)
llnames=arrays.files
print(llnames)

for x in llnames:
    vars()[x]=np.array(arrays[x])

i=np.lexsort((sku,oid))

for x in ['oid', 'sku', 'units', 'price', 'pah', 'tax', 'cost', 'status', 'shipped','ll_link']:
    vars()[x]=vars()[x][i]

#ORDER_NO        SKU     UNIT_PRICE      UNITS   PAH     TAX     UNIT_COST       LINE_SHIPDATE
orderno,skuno,shipdate = np.genfromtxt('/home/dmf/csv/skucodedetails/shipdate.txt',delimiter = '\t',skip_header = 1,usecols = [0,1,7],encoding = 'latin1',unpack = True,converters = {0: lambda s: int(s or 0), 1: lambda s: str(s or ''), 7: lambda s: int(s or 0)})

keep=np.intersect1d(orderno,oid)
i=np.isin(orderno,keep)
for x in ['orderno','skuno','shipdate']:
    vars()[x]=vars()[x][i]

skuno=np.searchsorted(parts[:,0],skuno)

i=np.lexsort((skuno,orderno))

for x in ['orderno','skuno','shipdate']:
    vars()[x]=vars()[x][i]

a=np.stack((oid,sku,shipped),axis=1)
a=np.hstack((a,np.zeros_like(a)))
b=np.stack((orderno,skuno,shipdate),axis=1)

a[i,:3].size==b.size

i=np.searchsorted(a[:,0],b[:,0])
x=np.unique(orderno,return_counts=1)[1]

y=[]
for z in np.nditer(x):
    y.append(np.arange(z))

y=np.concatenate(y)
i=i+y

a[i,3:]=b
shipped[i]=a[i,5]

np.savez_compressed('/home/dmf/python/npz/lldechiloshipdate.npz',oid=oid,sku=sku,units=units,price=price,pah=pah,tax=tax,cost=cost,status=status,statuskey=statuskey,shipped=shipped,parts=parts,oldskukey=oldskukey,ll_link=ll_link)

