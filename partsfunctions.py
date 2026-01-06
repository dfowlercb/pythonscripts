import numpy as np
from numpy.lib import recfunctions as rfn

def structured(x):
    dt=np.dtype([ ('sku','U21' ), ('cat' , 'U9' ), ('bg' ,'U21' ) , ('t1' ,'U2' ), ('t2' ,'U3' ), ('t3' ,'U2' ), ('catkey' ,'i4' ), ('bgkey' ,'i4' ), ('t1key' ,'i4' ), ('t2key' ,'i4' ), ('t3key' ,'i4' ) ])
    strucparts=rfn.unstructured_to_structured(x,dtype=dt)
    return(strucparts)

