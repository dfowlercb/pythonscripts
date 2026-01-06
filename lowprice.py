import numpy as np
from numpy.lib import recfunctions as rfn
from numpy import ma
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

arr=np.load('/home/dmf/nn/arr0.npz')
arr0=arr['arr0']
arr0['units']=ma.masked_values(arr0['units'],0)
arr0=arr0[arr0['cid'].argsort()]
idx0=np.unique(arr0['cid'],return_index=True)[1]
idx1=np.unique(arr0['cid'],return_counts=True)[1]
idx1=np.cumsum(idx1)
idx2=list(zip(idx0,idx1))

def weighted_ave(x,y):
   z=np.ma.average(x,weights=y)
   return(z)

lst0=[]
for x,y in idx2:
   lst0.append(weighted_ave(arr0['price'][x:y],arr0['units'][x:y])) 

aveprice=np.array(lst0)
aveprice=np.around(aveprice,2)
arr0=rfn.append_fields(arr0,'aveprice',aveprice[x[1]],usemask=False)
np.savez_compressed('/home/dmf/nn/arr0.npz',arr0=arr0,aveprice=aveprice)

import numpy as np
from numpy.lib import recfunctions as rfn
from numpy import ma
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

arr=np.load('/home/dmf/nn/arr0.npz')
arr0=arr['arr0']

arr1=arr0[['date','cid','aveprice','bu']]
year=arr1['date'].astype('datetime64[D]').astype('datetime64[Y]')
year=year.astype(int)+1970
arr1=rfn.append_fields(arr1,'year',year,usemask=False)
arr1=arr1[np.isfinite(arr1['aveprice'])==True]
arr1=arr1[arr1['aveprice']>0]

bins=np.array([4.5,6,7.5,9,10,11.5,13.5,17.0,24,250000])
pricebins=np.digitize(arr1['aveprice'],bins)
arr1=rfn.append_fields(arr1,'pricebins',pricebins,usemask=False)

i=np.where( (arr1['year']==2018) & (arr1['bu']==0) )
arr2=np.unique(arr1[['cid','aveprice','pricebins']][i])
output=np.bincount(arr2['pricebins'])
np.savetxt('/home/dmf/Documents/x.txt',output,delimiter='\t')

