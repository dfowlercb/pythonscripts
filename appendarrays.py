import numpy as np
from numpy.lib import recfunctions as rfn
arr_0=np.load('/home/dmf/python/npz/high2017_2022.npz')['arr_0']
arr_1=np.load('/home/dmf/python/npz/low2017_2022.npz')['arr_0']
print(arr_1.dtype.names)
delarr=np.array([arr_1.dtype.names])
idx=np.isin(delarr,np.array(['ORDER_NO','SKU','UNIT_COST']),invert=True)
delarr=delarr[idx]
arr_1=rfn.drop_fields(arr_1,delarr)
np.array_equal(arr_1['ORDER_NO'],arr_0['ORDER_NO'])
np.array_equal(arr_1['SKU'],arr_0['SKU'])
arr_0=rfn.append_fields(arr_0,'UNIT_COST',arr_1['UNIT_COST'],usemask=False)
arr_0[-10:]
