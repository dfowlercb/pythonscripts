import os
os.chdir('/home/dfowler/csv')
import numpy as np
from numpy.lib import recfunctions as rfn
from pyfunc.loads import load_orders as lo

def stack(arr0):
    arr1=lo(list0)
    x=rfn.stack_arrays((arr0,arr1),usemask=False)
    return(x)

from tuples.o2017_01 import list0
arr0=lo(list0)
from tuples.o2017_02 import list0
arr0=stack(arr0)
from tuples.o2017_03 import list0
arr0=stack(arr0)
from tuples.o2017_04 import list0
arr0=stack(arr0)
from tuples.o2017_05 import list0
arr0=stack(arr0)
from tuples.o2017_06 import list0
arr0=stack(arr0)
from tuples.o2017_07 import list0
arr0=stack(arr0)
from tuples.o2017_08 import list0
arr0=stack(arr0)
from tuples.o2017_09 import list0
arr0=stack(arr0)
from tuples.o2017_10 import list0
arr0=stack(arr0)
from tuples.o2017_11 import list0
arr0=stack(arr0)
from tuples.o2017_12 import list0
arr0=stack(arr0)

print(np.min(arr0['date']))
print(np.max(arr0['date']))
print(np.unique(arr0['date']).size)

np.savez_compressed('/home/dmf/python/npz/o2017.npz',arr0)
