import numpy as np
from numpy.lib import recfunctions as rfn

x=np.load('/home/dfowler/npz/bib.npz')['arr_0']
sales=x['UNIT_PRICE']*x['UNITS']
cogs=x['UNIT_COST']*x['UNITS']
margin=sales-cogs
date=x['ORDER_DATE'].astype('datetime64[D]')
year=x['ORDER_DATE'].astype('datetime64[D]').astype('datetime64[Y]')

x=rfn.append_fields(x,('SALES','COGS','MARGIN','DATE','YEAR'),(sales,cogs,margin,date,year),usemask=False)
x=rfn.drop_fields(x,('ORDER_TYPE','ORDER_DATE','FORM_PAY','BILLTO_ZIPCODE','BILLTO_STATE','NEW_CUST_FLAG','ENTRY_METHD','GENDER','MEMBER_CODE','LINE STATUS','LINE_SHIPDATE','CHUR_TAX_CID','CHUR_TAX_HID','TAX','UNIT_COST','HH_ID'))

y=np.load('/home/dfowler/npz/bibmid.npz')['arr_0']
sales=y['UNIT_PRICE']*y['UNITS']
cogs=y['UNIT_COST']*y['UNITS']
margin=sales-cogs
date=y['ORDER_DATE'].astype('datetime64[D]')
year=y['ORDER_DATE'].astype('datetime64[D]').astype('datetime64[Y]')

y=rfn.append_fields(y,('SALES','COGS','MARGIN','DATE','YEAR'),(sales,cogs,margin,date,year),usemask=False)
y=rfn.drop_fields(y,('ORDER_TYPE','ORDER_DATE','FORM_PAY','BILLTO_ZIPCODE','BILLTO_STATE','NEW_CUST_FLAG','ENTRY_METHD','GENDER','MEMBER_CODE','LINE STATUS','LINE_SHIPDATE','CHUR_TAX_CID','CHUR_TAX_HID','TAX','UNIT_COST','HH_ID'))

z=rfn.stack_arrays((x,y))

headers='CID,OID,SKU,PRICE,UNITS,PAH,SALES,COGS,MARGIN,DATE,YEAR'
np.savetxt('/home/dfowler/csv/bib.csv',z,delimiter=',',fmt='%i,%i,%s,%10.2f,%i,%10.2f,%10.2f,%10.2f,%10.2f,%s,%s',header=headers,comments='')

