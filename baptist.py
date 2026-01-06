import numpy as np
from numpy.lib import recfunctions as rfn

arr21=np.load('/home/dfowler/npz/h21.npz')['arr_0']
arr22=np.load('/home/dfowler/npz/h22.npz')['arr_0']
arr21=rfn.drop_fields(arr21,['HH_ID','CHUR_TAX_CID','CHUR_TAX_HID'])
arr22=rfn.drop_fields(arr22,['HH_ID','CHUR_TAX_CID','CHUR_TAX_HID'])

dt=np.dtype([ ('CID','i8'), ('BILLTO','U100'), ('SHIPTO','U50') ])
exempt21=np.loadtxt('/home/dfowler/csv/exempt21names.txt',delimiter='\t',dtype=dt,skiprows=1)
exempt22=np.loadtxt('/home/dfowler/csv/exempt22names.txt',delimiter='\t',dtype=dt,skiprows=1)
church=rfn.stack_arrays((exempt21,exempt22),usemask=False)
church=np.unique(church)

def keywordsearch(term,dbase):
    keyword=np.repeat(0,dbase.size)
    church=rfn.append_fields(dbase,'KEYWORD',keyword,usemask=False)
    idx=np.char.find(church['BILLTO'],term)
    church['KEYWORD'][np.where(idx>0)]=1
    print(np.unique(church['CID']).size)
    print(np.unique(church['CID'][church['KEYWORD']==1]).size)

keywordsearch('BAPTIST',church)
keywordsearch('METHODIST',church)
keywordsearch('CATHOLIC',church)
keywordsearch('ASSEMBL',church)
keywordsearch('LUTHERAN',church)
keywordsearch('PENTECOST',church)
church['BILLTO'][np.random.choice(church.shape[0],10,replace=False)].reshape(-1,1)

bib=np.load('/home/dfowler/npz/bib.npz')['arr_0']
idx=np.where(bib['ORDER_DATE']>=np.datetime64('2021-01-01').astype(int))
bib=bib[idx]
ORDER_NO=bib['ORDER_NO'].copy()
CM_ID=bib['CM_ID'].copy()
ORDER_DATE=bib['ORDER_DATE'].copy()
SKU=bib['SKU'].copy()
UNIT_PRICE=bib['UNIT_PRICE'].copy()
UNITS=bib['UNITS'].copy()
dt=np.dtype([('ORDER_NO','i8')])
bib=rfn.unstructured_to_structured(ORDER_NO.reshape(-1,1),dtype=dt)
bib=rfn.append_fields(bib,names=('CM_ID','ORDER_DATE','SKU','UNIT_PRICE','UNITS'),data=(CM_ID,ORDER_DATE,SKU,UNIT_PRICE,UNITS),usemask=False)

arr_0=rfn.stack_arrays((arr21,arr22,bib),usemask=False)
churchcid=np.unique(church['CID'])
idx=np.isin(arr_0['CM_ID'],churchcid)
arr_0=arr_0[idx]

exemptname=church.copy()
exemptname=rfn.drop_fields(exemptname,('OID','SHIPTO'))
exemptname=np.unique(exemptname)
exemptname=np.sort(exemptname,order='CID')

lst0=[]
for x in np.arange(1,exemptname['CID'].size):
    lst0.append(exemptname['CID'][x]!=exemptname['CID'][x-1])

lst0.insert(0,True)

exemptname=exemptname[np.array(lst0)]
idx=np.searchsorted(exemptname['CID'],arr_0['CM_ID'])
arr_0=rfn.append_fields(arr_0,'BILLTO',exemptname['BILLTO'][idx],usemask=False)

dt=np.dtype([('SKU','U21'),('PUB','U4')])
parts=np.loadtxt('/home/dfowler/csv/parts.txt',delimiter=',',dtype=dt,usecols=(6,10),encoding='latin1',skiprows=1)
parts=np.sort(parts,order='SKU')
idx=np.searchsorted(parts['SKU'],arr_0['SKU'])
arr_0=rfn.append_fields(arr_0,'PUB',parts['PUB'][idx],usemask=False)
arr_0=np.sort(arr_0,order='CM_ID')
del(arr21,arr22,bib,ORDER_NO,CM_ID,ORDER_DATE,SKU,UNIT_PRICE,UNITS,parts,idx,dt)

exemptname=church.copy()
exemptname=rfn.drop_fields(exemptname,('OID','SHIPTO'))
exemptname=np.unique(exemptname)
exemptname=np.sort(exemptname,order='CID')

lst0=[]
for x in np.arange(1,exemptname['CID'].size):
        lst0.append(exemptname['CID'][x]!=exemptname['CID'][x-1])

        lst0.insert(0,True)

        exemptname=exemptname[np.array(lst0)]
        idx=np.searchsorted(exemptname['CID'],arr_0['CM_ID'])
        arr_0=rfn.append_fields(arr_0,'BILLTO',exemptname['BILLTO'][idx],usemask=False)

idx=np.char.find(arr_0['BILLTO'],'BAPTIST')
arr_0[np.where(idx>=0)]

baptist=np.repeat(0,arr_0.size)
lifeway=np.repeat(0,arr_0.size)
arr_0=rfn.append_fields(arr_0,('BAPTIST','LIFEWAY'),(baptist,lifeway),usemask=False)

baptist_church=np.unique(church['CID'][np.where(church['BAPTIST']==1)])
idx=np.isin(arr_0['CM_ID'],baptist_church)
arr_0['BAPTIST'][idx]=1

idx=np.isin(arr_0['PUB'],np.array(['LFW','LFWW','LFWD','LFWX']))
arr_0['LIFEWAY'][idx]=1
idx=np.logical_and(arr_0['LIFEWAY']==1,arr_0['BAPTIST']==0)
np.unique(arr_0['CM_ID'][arr_0['BAPTIST']==0]).size
np.unique(arr_0['CM_ID'][idx]).size





