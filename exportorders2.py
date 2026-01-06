import numpy as np
from numpy.lib import recfunctions as rfn
x=np.load('/home/dfowler/npz/l2018_2021.npz')['arr_0']

SALES=x['UNITS']*x['UNIT_PRICE']
COGS=x['UNITS']*x['UNIT_COST']
DATE=x['ORDER_DATE'].astype('datetime64[D]')
YEAR=x['ORDER_DATE'].astype('datetime64[D]').astype('datetime64[Y]')
x=rfn.append_fields(x,('SALES','COGS','DATE','YEAR'),(SALES,COGS,DATE,YEAR),usemask=False)
x=np.sort(x,order=['DATE'])

np.savetxt('/home/dfowler/csv/pythonorders2.csv',x,delimiter=',',fmt='%i,%s,%s,%s,%s,%s,%s,%s,%s,%s,%10.2f,%10.2f,%10.2f,%i,%i,%i,%i,%i,%i,%i,%10.2f,%i,%10.2f,%10.2f,%s,%s',header='ORDER_NO, ORDER_TYPE, FORM_PAY, BILLTO_ZIPCODE, BILLTO_STATE, NEW_CUST_FLAG, ENTRY_METHD, GENDER, MEMBER_CODE, SKU, UNIT_COST, PAH, TAX, LINE STATUS, LINE_SHIPDATE, CM_ID, ORDER_DATE, HH_ID, CHUR_TAX_CID, CHUR_TAX_HID, UNIT_PRICE, UNITS, SALES, COGS, DATE, YEAR',comments='')


