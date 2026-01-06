import numpy as np
import csv
import os
import ast
from pyfunc.dateconvert import convertdate,convertdate2
import numpy.lib.recfunctions as rfn
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

demand = []

with open("/home/dfowler/csv/DEMAND_ORDERS_RPT.csv", "r") as f:
	contents = csv.reader(f)
	for c in contents:
		demand.append(c)

fields=['ORDER_DATE','ORD_TY','ORD_LY','SLS_TY','SLS_LY']
for f in fields:
    vars()[f]=np.zeros(0)

for d in demand:
    ORDER_DATE=np.append(ORDER_DATE,d[0])
    ORD_TY=np.append(ORD_TY,d[1])
    ORD_LY=np.append(ORD_LY,d[4])
    SLS_TY=np.append(SLS_TY,d[5])
    SLS_LY=np.append(SLS_LY,d[8])

indices=np.isin(ORDER_DATE,['DATE','WTD','MTD','YTD'],invert=True)
for f in fields:
    vars()[f]=vars()[f][indices]

for f in fields:
    vars()[f]=[ele.replace(",",'').replace("$","") for ele in vars()[f]]

ORDER_DATE=[convertdate(ele) for ele in ORDER_DATE]
ORD_TY=[int(ele) for ele in ORD_TY]
ORD_LY=[int(ele) for ele in ORD_LY]
SLS_TY=[float(ele) for ele in SLS_TY]
SLS_LY=[float(ele) for ele in SLS_LY]

report=list(zip(ORDER_DATE,ORD_TY,ORD_LY,SLS_TY,SLS_LY))

dt=np.dtype([ ('ORDER_DATE','M8[D]'), ('ORD_TY','i4'), ('ORD_LY','i4'), ('SLS_TY','f4'), ('SLS_LY','f4')  ])
report=np.array(report,dtype=dt)

high22=np.load('/home/dfowler/npz/h22.npz')['arr_0']
high23=np.load('/home/dfowler/npz/jan23hi.npz')['arr_0']
high_ord=rfn.stack_arrays((high22,high23))
sales=high_ord['UNITS']*high_ord['UNIT_PRICE']
high_ord=rfn.append_fields(high_ord,'SALES',sales,dtypes='f8',usemask=False)

bib22=np.load('/home/dfowler/npz/bib.npz')['arr_0']
bib23=np.load('/home/dfowler/npz/bibmid.npz')['arr_0']
bib_ord=rfn.stack_arrays((bib22,bib23))
names=np.array(bib_ord.dtype.names)
names=names[np.isin(names,np.array(['ORDER_NO','ORDER_DATE','UNITS','UNIT_PRICE','SHIP DATE','EXT_PRICE']),invert=True)]
bib_ord=rfn.drop_fields(bib_ord,names)
sales=bib_ord['EXT_PRICE']
bib_ord=rfn.append_fields(bib_ord,'SALES',sales,dtypes='f8',usemask=False)

def export_report(x):
    int_date=np.datetime64(x,'D').astype(int)
    rpt_amt=report[report['ORDER_DATE']==np.datetime64(x)]['SLS_TY'].item()
    i=np.where( high_ord['ORDER_DATE']==int_date) 
    hi_sls=np.sum(high_ord['SALES'][i])
    i=np.where(bib_ord['ORDER_DATE']==int_date)
    bib_sls=np.sum(bib_ord['SALES'][i])
    tot_amt=(hi_sls+bib_sls).item()
    dif_amt=tot_amt-rpt_amt
    output=np.array([rpt_amt,hi_sls,bib_sls,tot_amt,dif_amt]) 
    return(output)

export=np.empty(shape=(5,))
for x in np.arange(np.datetime64('2023-01-01'),np.datetime64('2023-01-25')):
    y=export_report(x)
    export=np.vstack((export,y))

np.savetxt('/home/dfowler/documents/demand_reconcile.txt',export[1:],delimiter=',',fmt='%03.2f')



