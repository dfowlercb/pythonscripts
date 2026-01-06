import numpy as np
from numpy.lib import recfunctions as rfn
x=np.load('/home/dfowler/npz/l2021_2023a.npz')['arr_0']
idx=np.where(x['CHUR_TAX_CID']!=0)
x=x[idx]

parts=np.loadtxt('/home/dfowler/csv/parts.txt',dtype=[('part' ,'U13' ), ('cat' ,'U9' )],delimiter=',',encoding='latin1',usecols=(6,11),skiprows=1)
parts=np.sort(parts,order='part')
idx=np.searchsorted(parts['part'],x['SKU'])
CAT=parts['cat'][idx]
DATE=x['ORDER_DATE'].astype('datetime64[D]')
tiers=np.char.split(CAT,sep='-')
t1 = np.array([value[0] for value in tiers])
t2 = np.array([value[1] for value in tiers])
t3 = np.array([value[2] for value in tiers])
x=rfn.append_fields(x,('DATE','T1','T2','T3'),(DATE,t1,t2,t3),usemask=False)
idx=np.where(x['T1']=='05')
x=x[idx]

x=rfn.drop_fields(x,('ORDER_NO','ORDER_TYPE','ORDER_DATE','FORM_PAY','BILLTO_ZIPCODE','BILLTO_STATE','NEW_CUST_FLAG','ENTRY_METHD','GENDER','MEMBER_CODE','LINE_STATUS','LINE_SHIPDATE','CM_ID','HH_ID','CHUR_TAX_CID','CHUR_TAX_HID','PAH','TAX','LINE STATUS'))

np.savetxt('/home/dfowler/csv/piper.txt',x,fmt='%s %10.2f %10.2f %i %s %s %s %s')
