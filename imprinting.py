import numpy as np
from scipy import sparse
import os
from numpy import genfromtxt
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

#LOAD FILES
os.chdir('/home/dmf/python/npz')
 #order level   
arrays=np.load('oldechilo.npz',allow_pickle=1)

oidol=arrays['oidol']
date=arrays['date']

 #line level
arrays=np.load('lldechilo.npz',allow_pickle=1)
oid=arrays['oid']
sku=arrays['sku']
units=arrays['units']
price=arrays['price']
parts=arrays['parts']
ll_link=arrays['ll_link']

#date magic
date = date.astype('datetime64[D]')
years = date.astype('datetime64[Y]') 
months= (date.astype('datetime64[M]')-date.astype('datetime64[Y]')+1).astype('int64')
weeks = (date.astype('datetime64[W]')-date.astype('datetime64[Y]')+1).astype('int64')
weekdays = (date.astype('datetime64[D]').astype('int64') -4) % 7 +1
monthdays =  date.astype('datetime64')-date.astype('datetime64[M]') + 1
yeardays = date.astype('datetime64')-date.astype('datetime64[Y]') + 1

years=np.unique(years,return_inverse=1)
months=np.unique(months,return_inverse=1)
weeks=np.unique(weeks,return_inverse=1)
weekdays=np.unique(weekdays,return_inverse=1)
monthdays=np.unique(monthdays,return_inverse=1)
yeardays=np.unique(yeardays,return_inverse=1)

#find bibles 
bibles=np.where(parts[:,3]=='02')
i=np.isin(sku,bibles)

#matrices
data=units[i]*price[i]

 #annual bible sales
row=np.repeat(0,data.size)
col=years[1][ll_link][i]
A=sparse.coo_matrix((data,(row,col)))
A.toarray()

 #weekly bible sales
row=weeks[1][ll_link][i]
row[np.where(row>51)[0]]=51
col=years[1][ll_link][i]
A=sparse.coo_matrix((data,(row,col)))
A.toarray()

os.chdir('/home/dmf/Documents')
np.savetxt('bibles.txt',A.toarray(),delimiter=',',fmt='%f')

#LINE CHART
x=range(1,53)
y2021=np.transpose(A.toarray())[24]

plt.plot(x,y2021)
plt.show()

