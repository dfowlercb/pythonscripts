#replace TWICE ,, with ,0, in ministryflags2.csv MUST DO TWO TIMES
import numpy as np
from numpy.lib import recfunctions as rfn
from datetime import datetime as dt
x=np.load('/home/dfowler/npz/l2022_2025a.npz')['arr_0']
x=np.sort(x,order='CM_ID')

conv={
        #13: lambda s: np.datetime64(dt.strptime(s, '%m %d %Y'),'D'),
        14: lambda s: int(float(s)) or 0,
        15: lambda s: int(float(s)) or 0
}

flag=np.loadtxt('/home/dfowler/csv/ministryflags2.csv',dtype=[('CID' ,'i8' ), ('FLAG' ,'i8' ),('UNSUBSCRIBE','U20'),('RECENCY','U10'),('FREQUENCY','i8'),('MONETARY','f8'),('MARKETPLACE','U20')],delimiter=',',encoding='latin1',usecols=(0,2,11,13,14,15,16),skiprows=1,converters=conv,comments='None')
flag=np.unique(flag)
flag=np.sort(flag,order='CID')

idx=np.isin(x['CM_ID'],flag['CID'])
np.unique(idx,return_counts=True)
x=x[idx]
idx=np.isin(x['CM_ID'],flag['CID'])
np.unique(idx,return_counts=True)

idx=np.searchsorted(flag['CID'],x['CM_ID'])

FLAG=flag['FLAG'][idx]
SALES=x['UNITS']*x['UNIT_PRICE']
COGS=x['UNITS']*x['UNIT_COST']
DATE=x['ORDER_DATE'].astype('datetime64[D]')
YEAR=x['ORDER_DATE'].astype('datetime64[D]').astype('datetime64[Y]')

x=rfn.append_fields(x,('FLAG','SALES','COGS','DATE','YEAR'),(FLAG,SALES,COGS,DATE,YEAR),usemask=False)
x=rfn.drop_fields(x,('ORDER_TYPE','ORDER_DATE','FORM_PAY','BILLTO_ZIPCODE','BILLTO_STATE','NEW_CUST_FLAG','ENTRY_METHD','GENDER','MEMBER_CODE','LINE STATUS','LINE_SHIPDATE','CHUR_TAX_CID','CHUR_TAX_HID','TAX','UNIT_COST'))

idx=np.where( (x['DATE']>=np.datetime64('2023-01-01')) & (x['DATE']<=np.datetime64('2023-12-31')) )
y=x[idx]
np.bincount(y['FLAG'],weights=y['SALES'])
np.sum(np.bincount(y['FLAG'],weights=y['SALES']))

np.savetxt('/home/dfowler/csv/church.csv',x,delimiter=',',fmt='%i,%s,%10.2f,%i,%i,%10.2f,%i,%i,%10.2f,%10.2f,%s,%s',header='OID,SKU,PAH,CID,HID,PRICE,UNITS,FLAG,SALES,COGS,DATE,YEAR',comments='')

tri=np.loadtxt('/home/dfowler/csv/cbtriappend2.txt',dtype=[('ADID' , 'i8'), ('CID' ,'i8' ), ('DENOMNAME' ,'U50' ),('DENOMGROUP' ,'U24'), ('RECENCY','U10'), ('FREQUENCY','i8'),('MONETARY','i8'), ('UNSUBSCRIBE','U24'),('MARKETPLACE','U24'),('FLAG','i8')],delimiter='\t',encoding='latin1',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9))
tri['DENOMGROUP'][tri['DENOMGROUP']=='']='UNKNOWN'
tri=np.sort(tri,order='CID')

idx=np.isin(x['CM_ID'],tri['CID'])
denom=x[idx].copy()
idx=np.isin(x['CM_ID'],tri['CID'],invert=True)
nodenom=x[idx].copy()

denom=np.sort(denom,order='CM_ID')
idx=np.searchsorted(tri['CID'],denom['CM_ID'])
ADID=tri['ADID'][idx]
DENOMNAME=tri['DENOMNAME'][idx]
DENOMGROUP=tri['DENOMGROUP'][idx]
RECENCY=tri['RECENCY'][idx]
RECENCY=np.array([np.datetime64(s,'D') for s in RECENCY])
FREQUENCY=tri['FREQUENCY'][idx]
MONETARY=tri['MONETARY'][idx]
MONETARY=MONETARY.astype(float)
UNSUBSCRIBE=tri['UNSUBSCRIBE'][idx]
MARKETPLACE=tri['MARKETPLACE'][idx]

denom=rfn.append_fields(denom,('ADID','DENOMNAME','DENOMGROUP','RECENCY','FREQUENCY','MONETARY','UNSUBSCRIBE','MARKETPLACE'),(ADID,DENOMNAME,DENOMGROUP,RECENCY,FREQUENCY,MONETARY,UNSUBSCRIBE,MARKETPLACE),usemask=False)

flag['RECENCY'][np.where(flag['RECENCY']=='0')]='12 17 2014'
recency2=np.array([dt.strptime(s, '%m %d %Y') for s in flag['RECENCY']]).astype(np.datetime64)
recency2=np.array([np.datetime64(s,'D') for s in recency2])

nodenom=np.sort(nodenom,order='CM_ID')
idx=np.searchsorted(flag['CID'],nodenom['CM_ID'])
length=len(nodenom)
ADID=np.repeat(-1,length)
DENOMNAME=np.repeat('',length)
DENOMGROUP=np.repeat('',length)
RECENCY=recency2[idx]
FREQUENCY=flag['FREQUENCY'][idx]
MONETARY=flag['MONETARY'][idx]
UNSUBSCRIBE=flag['UNSUBSCRIBE'][idx]
MARKETPLACE=flag['MARKETPLACE'][idx]
FLAG=flag['FLAG'][idx]

DENOMNAME=DENOMNAME.astype(np.dtype('U50'))
DENOMGROUP=DENOMGROUP.astype(np.dtype('U24'))
UNSUBSCRIBE=UNSUBSCRIBE.astype(np.dtype('U24'))
MARKETPLACE=MARKETPLACE.astype(np.dtype('U24'))

nodenom=rfn.append_fields(nodenom,('ADID','DENOMNAME','DENOMGROUP','RECENCY','FREQUENCY','MONETARY','UNSUBSCRIBE','MARKETPLACE'),(ADID,DENOMNAME,DENOMGROUP,RECENCY,FREQUENCY,MONETARY,UNSUBSCRIBE,MARKETPLACE),usemask=False)

y=rfn.stack_arrays((denom,nodenom),usemask=False)

np.savetxt('/home/dfowler/csv/church2.txt',y,delimiter='|',fmt='%i|%s|%10.2f|%i|%i|%10.2f|%i|%i|%10.2f|%10.2f|%s|%s|%i|%s|%s|%s|%i|%i|%s|%s',header='OID|SKU|PAH|CID|HID|PRICE|UNITS|FLAG|SALES|COGS|DATE|YEAR|ADID|DENOMNAME|DENOMGROUP|RECENCY|FREQUENCY|MONETARY|UNSUBSCRIBE|MARKETPLACE',comments='')



#import numpy as np
#from numpy.lib import recfunctions as rfn
#x=np.load('/home/dfowler/npz/bib.npz')['arr_0']
#x=x[x['LINE STATUS']==2]
#SALES=x['UNITS']*x['UNIT_PRICE']
#COGS=x['UNITS']*x['UNIT_COST']
#DATE=x['ORDER_DATE'].astype('datetime64[D]')
#YEAR=x['ORDER_DATE'].astype('datetime64[D]').astype('datetime64[Y]')
#
#x=rfn.append_fields(x,('SALES','COGS','DATE','YEAR'),(SALES,COGS,DATE,YEAR),usemask=False)
#x=rfn.drop_fields(x,('ORDER_TYPE','ORDER_DATE','FORM_PAY','BILLTO_ZIPCODE','BILLTO_STATE','NEW_CUST_FLAG','ENTRY_METHD','GENDER','MEMBER_CODE','LINE STATUS','LINE_SHIPDATE','HH_ID','CHUR_TAX_CID','CHUR_TAX_HID','TAX','UNIT_COST'))

#idx=np.where( (x['DATE']>=np.datetime64('2023-02-01')) & (x['DATE']<=np.datetime64('2024-01-31')) )
#np.sum(x['SALES'][idx])
#
#np.savetxt('/home/dfowler/csv/bib.csv',x,delimiter=',',fmt='%i,%i,%s,%10.2f,%i,%10.2f,%10.2f,%10.2f,%s,%s' ,header='CID,OID,SKU,PRICE,UNITS,PAH,SALES,COGS,DATE,YEAR',comments='')


