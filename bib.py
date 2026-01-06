import numpy as np
from datetime import datetime
def convertdate(x):
    x=np.datetime64(datetime.strptime(x,'%m/%d/%y').date())
    return x

#CM_ID	0
#ORDER_NO	1
#ORDER_DATE	2
#SKU_CODE	3
#SKU	4
#TITLE	5
#UNIT_PRICE	6
#UNITS	7
#GENDER	8
#ORDER_TYPE	9
#MEMBER_CODE	10
#BUSINESS_FLAG	11
#CATALOG_CODE	12
#RENTAL_NO	13
#BILLTO_ZIPCODE	14
#PAH	15
#TAX	16
#EXT_PRICE	17
#NEW_CUST.FLAG	18
#KEYCODE	19
#PROMO CODE	20
#FLOWCATALOG	21
#FLOWNAME	22
#CATEGORY	23
#UNIT_COST	24
#FORM_PAY	25
#BILLTO STATE	26
#ENTRY_METHOD	27
#HOUSEHOLD #	28
#LINE STATUS	29

dt=np.dtype([ ('oid','int'), ('date','M8[D]'), ('price','f4' ), ('units','i4' ) ])
bib=np.genfromtxt('/home/dmf/csv/bib.txt',usecols=(1,2,6,7),skip_header=1,dtype=dt,delimiter='\t',converters={2: convertdate},encoding='latin1')


arr1=np.genfromtxt('/home/dmf/csv/bib.txt',usecols=1,skip_header=1,dtype='int')
arr2=np.genfromtxt('/home/dmf/csv/bib.txt',usecols=1,skip_header=1,dtype='int')

