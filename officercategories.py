import numpy as np
import numpy.lib.recfunctions as rf
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})


ol=np.load('/home/dmf/python/npz/oljulhilo.npz')
ll=np.load('/home/dmf/python/npz/lljulhilo.npz')
date=ol['date']
oid=ll['oid']
sku=ll['sku']
units=ll['units']
price=ll['price']
cost=ll['cost']
parts=ll['parts']
ll_link=ll['ll_link']

sales=units*price
cost=units*cost
shipdate=ll['shipped']
shipmonthend=ll['shipped']

monthend=np.genfromtxt('/home/dmf/csv/month_end_dates.csv',dtype='int',names=True)

def monthendconvert(date2convert):
    for x in np.arange(len(monthend)):    
        idx=np.where( (date2convert>=monthend['start'][x]) & (date2convert<=monthend['end'][x] ) )
        date2convert[idx]=monthend['monthend'][x]
    
    return date2convert

shipmonthend=monthendconvert(shipmonthend)


dt=np.dtype([ ('oid','i4'), ('date','i4'), ('sku','i4'), ('price','f4'), ('sales','f4'), ('cost','f4'), ('shipdate','i4'), ('shipmonthend','i4') ])
arr1=np.array([oid,date[ll_link],sku,price,sales,cost,shipdate,shipmonthend]).T
arr1=rf.unstructured_to_structured(arr1,dtype=dt)
arr2=arr1.copy()
arr2=arr1[arr1['shipdate']==np.datetime64('2022-07-15').astype(int)]

#exclusions
 #exclude payroll lines
arr1=arr1[arr1['sku']!=1776651]
 #exclude free items
arr1=arr1[arr1['price']>0]
 #exclude reships
#arr1[arr1['sku']==1776710]

dt=np.dtype([ ('OID','i4')])
itdata=np.genfromtxt('/home/dmf/csv/0715.csv',delimiter=',',usecols=(0),dtype=dt,skip_header=1)
dt=np.dtype([('OID', 'i4'), ('CID', 'i4'), ('DATE', 'D'), ('HHID', 'i4'), ('CHURTAXCID', 'i4'), ('CHURTAXHID', 'i4'), ('SKU', 'U12'), ('PRICE', 'f4'), ('UNITS', 'i4'), ('ORDERTYPE', 'U1'), ('FORMPAY', 'U2'), ('BILLTOZIPCODE', 'U5'), ('BILLTOSTATE', 'U2'), ('NEWCUSTFLAG', 'U1'), ('ENTRYMETHD', 'U12'), ('GENDER', 'U1'), ('MEMBER_CODE', 'U1'), ('UNITCOST', 'f4'), ('PAH', 'f4'), ('TAX', 'f4'), ('LINE_STATUS', 'i4'), ('LINESHIPDATE', 'i4')])

i0=np.isin(arr1['oid'],itdata['OID'])
i1=np.isin(itdata['OID'],arr1['oid'])


#bib=np.genfromtxt('/home/dmf/csv/bibskucodedetails.txt',delimiter='\t',dtype=dt,skip_header=1)
bib=np.genfromtxt('/home/dmf/csv/bibskucodedetails.txt',delimiter='\t',dtype=dt,skip_header=1,converters = {2: lambda s: np.datetime64(s,'D')})


np.savetxt('/home/dmf/csv/monthend.txt',arr2)

np.sum(arr1[ (arr1['shipdate']>=np.datetime64('2022-07-31').astype(int)) & (arr1['shipdate']<=np.datetime64('2022-08-01').astype(int))]['sales'])

