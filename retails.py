import numpy as np
import os
from numpy.lib import recfunctions as rfn
from pyfunc_scripts.loads import load_po
from pyfunc_scripts.po import retail,month_lookup
#np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

os.chdir('/home/dmf/csv/')
#load and combine purchase orders, skus converted to numbers based on current parts file
for file in ('06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22'):
    globals()["".join(['po',file])]=load_po(file) 

po=rfn.stack_arrays((po06,po07,po08,po09,po10,po11,po12,po13,po14,po15,po16,po17,po18,po19,po20,po21,po22),usemask=False)
np.savez_compressed('/home/dmf/python/npz/po.npy',po)

#generate po sku, dates and retails
poskus,podates,poretails=retail(po)

#lookup skus for targeted date range
startdate='2016-01-01'
stopdate='2017-01-01'
fmth='sep'
export=month_lookup(startdate,stopdate,fmth,poskus,podates,poretails)

#match retails to hilo files
import numpy as np
arrays=np.load('/home/dmf/python/npz/retails_2022-07-01_2022-09-01.npz')
mretail=arrays['mretail']

ll=np.load('/home/dmf/python/npz/llaughilo.npz')
sku=ll['sku']
link=ll['ll_link']

ol=np.load('/home/dmf/python/npz/olaughilo.npz')
date=ol['date']
date=date[link]
date=date.astype('datetime64[D]').astype('datetime64[M]').astype(int)



