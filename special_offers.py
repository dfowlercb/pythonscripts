import numpy as np
from numpy.lib import recfunctions as rfn

arr18=np.load('/home/dmf/python/npz/o2018.npz')['arr_0']
arr19=np.load('/home/dmf/python/npz/o2019.npz')['arr_0']
arr20=np.load('/home/dmf/python/npz/o2020.npz')['arr_0']
arr21=np.load('/home/dmf/python/npz/o2021.npz')['arr_0']
arr22=np.load('/home/dmf/python/npz/o2022.npz')['arr_0']
arr0=rfn.stack_arrays((arr18,arr19,arr20,arr21,arr22))
del(arr18,arr19,arr20,arr21,arr22)
oid=np.copy(arr0['oid'])
promocode=np.copy(arr0['promocode'])
del(arr0)
oidpromocode=np.vstack((oid,promocode))
oidpromocode=np.unique(oidpromocode,axis=1)

#add a '' NOPROMOCODE row to beginning of text file and remove any duplicates
arr0=np.loadtxt('/home/dmf/csv/special_offers.csv',usecols=(0,1),delimiter='\t',dtype='str',skiprows=1)
arr0[0,0]=''
arr0=rfn.unstructured_to_structured(arr0)
arr0.dtype.names=('OFFER','TYPE')
arr0=np.sort(arr0,order='OFFER')
idx=np.searchsorted(arr0['OFFER'],oidpromocode[1])
promotype=arr0['TYPE'][idx]
oidpromocode=np.vstack((oidpromocode,promotype))
np.savetxt('/home/dmf/Documents/specialoffers.txt',oidpromocode.T,delimiter='\t',fmt='%s')
