import numpy as np
import numpy.lib.recfunctions as rf
from datetime import datetime
from dateconvert import convertdate
np.set_printoptions(suppress=True)

ll=np.load('/home/dmf/python/npz/llhigh.npz')
parts=ll['parts']
def convertpart(x):
    y=np.searchsorted(parts[:,0],x)
    return(y)

dt=np.dtype([('date','M8[D]'),('sku','i4'),('units','i4'),('retail','f4')])
po18=np.genfromtxt('/home/dmf/csv/podata2018.txt',usecols=[3,12,20,25],delimiter='\t',encoding='latin1',dtype=dt,converters={3:convertdate,12:convertpart},skip_header=1)
po19=np.genfromtxt('/home/dmf/csv/podata2019.txt',usecols=[3,12,20,25],delimiter='\t',encoding='latin1',dtype=dt,converters={3:convertdate,12:convertpart},skip_header=1)
po20=np.genfromtxt('/home/dmf/csv/podata2020.txt',usecols=[3,12,20,25],delimiter='\t',encoding='latin1',dtype=dt,converters={3:convertdate,12:convertpart},skip_header=1)
po21=np.genfromtxt('/home/dmf/csv/podata2021.txt',usecols=[3,12,20,25],delimiter='\t',encoding='latin1',dtype=dt,converters={3:convertdate,12:convertpart},skip_header=1)
po22=np.genfromtxt('/home/dmf/csv/podata2022.txt',usecols=[3,12,20,25],delimiter='\t',encoding='latin1',dtype=dt,converters={3:convertdate,12:convertpart},skip_header=1)
po=np.hstack((po18,po19,po20,po21,po22))

from numba import jit
@jit(nopython=True,parallel=True,nogil=True,fastmath=True)
def retailbymonth(date2,sku):
    y=np.array([date - date2 for date in podate[po['sku']==sku]])
    for i in range(len(y)):
            if y[i]<0:
                y[i]=y[i]*-1
        
    z=po[po['sku']==sku]['retail'][np.where(y==np.min(y))[0][0].item()]
    return(z)

vretailbymonth=np.vectorize(retailbymonth)

ol=np.load('/home/dmf/python/npz/olhigh.npz')
date=ol['date']
ll=np.load('/home/dmf/python/npz/llhigh.npz')
sku=ll['sku']
units=ll['units']
price=ll['price']
link=ll['ll_link']
parts=ll['parts']
arr1=np.array([sku, price, units] ).T
dt=np.dtype([ ('sku','i4' ), ('price' ,'f4' ), ('units' ,'i4' ) ])
arr1=rf.unstructured_to_structured(arr1,dtype=dt)
arr1=rf.append_fields(arr1,'date',date[link],dtypes='M8[D]',usemask=False)
arr1=arr1[arr1['date']>=np.datetime64('2018-01-01')]
arr1=arr1[np.isin(arr1['sku'],po['sku'])]

podate=po['date'].astype(int)
orderdate=arr1['date'].astype(int)

retail=vretailbymonth(date2=orderdate,sku=arr1['sku'])


