import numpy as np
import numpy.lib.recfunctions as rf

def convparts(arr):
    dt=np.dtype([('SKU','U13'),('CAT','U9'),('BG','U17'),('T1','U2'),('T2','U2'),('T3','U2'),('CATNUM','i4'),('BGNUM','i4'),('T1NUM','i4'),('T2NUM','i4'),('T3NUM','i4')  ])
    arr=rf.unstructured_to_structured(arr,dtype=dt)
    return(arr)

