import numpy as np
import re
import csv
import os
import numpy.lib.recfunctions as rfn
from pyfunc.dateconvert import convertdate, convertdate2
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

os.system('cp /home/dmf/Downloads/DAILY_SHIPPING_RPT_11_01_2022_TO_11_30_2022.csv /home/dmf/csv/DAILY_SHIPPING_RPT.csv')
 
shipped = []
with open("/home/dmf/csv/DAILY_SHIPPING_RPT.csv", "r") as f:
	contents = csv.reader(f)
	for c in contents:
		shipped.append(c)

for i in np.arange(len(shipped)):
    shipped[i]=[re.sub("[#,$,%, ,]","",x) for x in shipped[i]]

idx=[]
for s in shipped:
    idx.append(len(s)>1)

shipped=np.array(shipped)[idx]

fields=['date','sales']
for f in fields:
    vars()[f]=np.zeros(0)

for s in shipped:
    date=np.append(date,s[0])
    sales=np.append(sales,s[11])

date=date[1:-1]
date=np.array([convertdate(x) for x in date])
sales=sales[1:-1].astype(float)

report=rfn.unstructured_to_structured(np.array(date.reshape(-1,1)),names=['date'])
report=rfn.append_fields(report,'SALES',sales,usemask=False)

high=np.load('/home/dmf/python/npz/high2017_2022.npz')['arr_0']
low=np.load('/home/dmf/python/npz/low2017_2022.npz')['arr_0']
high=np.sort(high,order=('ORDER_NO','SKU'))
np.array_equal(high['ORDER_NO'],low['ORDER_NO'])
np.array_equal(high['SKU'],low['SKU'])

payroll=np.unique(high['ORDER_NO'][high['SKU']=='PAYROLL'])
high=high[np.isin(high['ORDER_NO'],payroll,invert=True)]

sales=high['UNITS']*high['UNIT_PRICE']
high=rfn.append_fields(high,('SALES','SHIP_DATE'),(sales,low['LINE_SHIPDATE']),dtypes=('f8','i8'),usemask=False)
high=high[high['ORDER_DATE']>=np.datetime64('2021-01-01').astype(int)]

bib=np.load('/home/dmf/python/npz/bib.npz')['arr_0']
sales=bib['UNIT_PRICE']*bib['UNITS']
bib=rfn.append_fields(bib,'SALES',sales,usemask=False)

def printdiff(x):
    dateref=x
    reportamount=report[report['date']==np.datetime64(dateref)]['SALES'].item()
    idx=high['SHIP_DATE']==np.datetime64(dateref).astype(int)
    filehilo=np.sum(high['SALES'][idx])
    filebib=np.sum(bib['SALES'][idx])
    fileamount=(filehilo+filebib).item()
    diffamount=int(fileamount-reportamount)
    print('The report amount is:'+str(int(reportamount)))
    print('The file amount is:'+str(int(fileamount)))
    print('The file amount consists of hilo:'+str(filehilo)+'and bib:'+str(filebib))
    print('The difference between the report amount and the file amount is:' + str(diffamount))

printdiff('2022-11-02')

unqoid=np.unique(oidol[i],return_counts=True)
test=np.genfromtxt('/home/dmf/csv/ordershiptest.csv',delimiter=',',skip_header=1,usecols=(0,1),dtype=[('oid','i4'),('lines','i4')])
np.sort(test)
np.array_equal(test['oid'],unqoid[0])
np.array_equal(test['lines'],unqoid[1])
x=np.array(test['lines']==unqoid[1])
test[np.where(x==False)]
unqoid[1][np.where(x==False)]
