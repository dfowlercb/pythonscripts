import numpy as np
from pyfunc.arrfunc import stackarr

ftype='l'
mid="no"
savname=ftype+'2018_2021'

fyears=('18','19','20','21')
#fyears=('1997_2001','2002_2006','2007_2011','2012_2016','2017_2022')

if ftype!='bib':
    arr_0=np.load('/home/dfowler/npz/'+ftype+fyears[0]+'.npz')['arr_0']
    [arr_0:=stackarr(arr_0,ftype+x) for x in fyears[1:]] 
    arr_0=np.sort(arr_0,order=('ORDER_NO','SKU'))
else:
    arr_0=np.load('/home/dfowler/npz/bib.npz')['arr_0']

if ftype=='h':
    print(np.unique(arr_0['ORDER_DATE'].astype('datetime64[D]').astype('datetime64[Y]')))

if mid=='yes':
    arr_0=stackarr(arr_0,'mid'+ftype)

np.savez_compressed('/home/dfowler/npz/'+savname+'.npz',arr_0)


