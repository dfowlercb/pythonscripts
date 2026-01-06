import numpy as np
from numpy.lib import recfunctions as rfn

arr_97=np.load('/home/dfowler/npz/hi1997_2001.npz')['arr_0']
arr_02=np.load('/home/dfowler/npz/hi2002_2006.npz')['arr_0']
arr_07=np.load('/home/dfowler/npz/hi2007_2011.npz')['arr_0']
arr_12=np.load('/home/dfowler/npz/hi2012_2016.npz')['arr_0']
arr_17=np.load('/home/dfowler/npz/hi2017_2022.npz')['arr_0']
arr_0=rfn.stack_arrays((arr_97,arr_02),usemask=False)
arr_0=rfn.stack_arrays((arr_0,arr_07),usemask=False)
arr_0=rfn.stack_arrays((arr_0,arr_12),usemask=False)
arr_0=rfn.stack_arrays((arr_0,arr_17),usemask=False)
del(arr_97,arr_02,arr_07,arr_12,arr_17)

dt=np.dtype([ ('CID','i8'), ('BU' ,'i8')])
bu=np.loadtxt('/home/dfowler/documents/cidbudec22.txt',delimiter='\t',skiprows=1,dtype=dt)
bu=np.sort(bu,order='CID')
idx=np.searchsorted(bu['CID'],arr_0['CM_ID'])
BU=bu['BU'][idx]
arr_0=rfn.append_fields(arr_0,'BU',BU,usemask=False)
arr_0=arr_0[arr_0['BU']==1]

year=arr_0['ORDER_DATE'].astype('datetime64[D]').astype('datetime64[Y]').astype(int)
arr_0=rfn.append_fields(arr_0,'YEAR',year,usemask=False)

sales=arr_0['UNITS']*arr_0['UNIT_PRICE']
arr_0=rfn.append_fields(arr_0,'SALES',sales,usemask=False)

for idx in np.arange(27,53):
    print(np.unique(arr_0['CM_ID'][arr_0['YEAR']==idx]).size)

for idx in np.arange(27,53):
    print(np.sum(arr_0['SALES'][arr_0['YEAR']==idx]))

