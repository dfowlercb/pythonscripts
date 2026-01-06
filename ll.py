#12.2.22.30 SKU,ORDER.NO,UNIT.PRICE,UNITS,UNIT.COST,PAH,TAX,LINE.STATUS
import os
from numpy import genfromtxt
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})
os.chdir('/home/dmf/csv/skucodedetails')

sku=np.empty(shape=0)
oid=np.empty(shape=0)
price=np.empty(shape=0)
units=np.empty(shape=0)
cost=np.empty(shape=0)
pah=np.empty(shape=0)
tax=np.empty(shape=0)
status=np.empty(shape=0)

for i in range (1997,2012):
    print(i)
    sku=np.hstack((sku,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='object',delimiter='\t',encoding='latin1',usecols=3,skip_header=1,converters={3: lambda s: str(s or '')})))
    oid=np.hstack((oid,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='int64',delimiter='\t',encoding='latin1',usecols=1,skip_header=1,converters={1: lambda s: int(s or 0)})))
    price=np.hstack((price,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='float64',delimiter='\t',encoding='latin1',usecols=4,skip_header=1,converters={4: lambda s: float(s or 0.)})))
    units=np.hstack((units,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='int64',delimiter='\t',encoding='latin1',usecols=5,skip_header=1,converters={5: lambda s: int(s or 0)})))
    pah=np.hstack((pah,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='float64',delimiter='\t',encoding='latin1',usecols=8,skip_header=1,converters={8: lambda s: float(s or 0.)})))
    tax=np.hstack((tax,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='float64',delimiter='\t',encoding='latin1',usecols=9,skip_header=1,converters={9: lambda s: float(s or 0.)})))
    cost=np.hstack((cost,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='float64',delimiter='\t',encoding='latin1',usecols=11,skip_header=1,converters={11: lambda s: float(s or 0.)})))
    status=np.hstack((status,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='object',delimiter='\t',encoding='latin1',usecols=14,skip_header=1,converters={14: lambda s: str(s or '')})))

for i in range (2012,2022):
    print(i)
    sku=np.hstack((sku,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='object',delimiter='\t',encoding='latin1',usecols=4,skip_header=1,converters={4: lambda s: str(s or '')})))
    oid=np.hstack((oid,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='int64',delimiter='\t',encoding='latin1',usecols=1,skip_header=1,converters={1: lambda s: int(s or 0)})))
    price=np.hstack((price,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='float64',delimiter='\t',encoding='latin1',usecols=6,skip_header=1,converters={6: lambda s: float(s or 0.)})))
    units=np.hstack((units,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='int64',delimiter='\t',encoding='latin1',usecols=7,skip_header=1,converters={7: lambda s: int(s or 0)})))
    pah=np.hstack((pah,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='float64',delimiter='\t',encoding='latin1',usecols=15,skip_header=1,converters={15: lambda s: float(s or 0.)})))
    tax=np.hstack((tax,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='float64',delimiter='\t',encoding='latin1',usecols=16,skip_header=1,converters={16: lambda s: float(s or 0.)})))
    cost=np.hstack((cost,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='float64',delimiter='\t',encoding='latin1',usecols=24,skip_header=1,converters={24: lambda s: float(s or 0.)})))
    status=np.hstack((status,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='object',delimiter='\t',encoding='latin1',usecols=29,skip_header=1,converters={29: lambda s: str(s or '')})))

#INTRAMONTH APPEND from skd
    sku=np.hstack((sku,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='object',delimiter='\t',encoding='latin1',usecols=1,skip_header=1,converters={1: lambda s: str(s or '')})))
    oid=np.hstack((oid,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='int64',delimiter='\t',encoding='latin1',usecols=2,skip_header=1,converters={2: lambda s: int(s or 0)})))
    price=np.hstack((price,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='float64',delimiter='\t',encoding='latin1',usecols=3,skip_header=1,converters={3: lambda s: float(s or 0.)})))
    units=np.hstack((units,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='int64',delimiter='\t',encoding='latin1',usecols=4,skip_header=1,converters={4: lambda s: int(s or 0)})))
    pah=np.hstack((pah,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='float64',delimiter='\t',encoding='latin1',usecols=5,skip_header=1,converters={5: lambda s: float(s or 0.)})))
    tax=np.hstack((tax,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='float64',delimiter='\t',encoding='latin1',usecols=6,skip_header=1,converters={6: lambda s: float(s or 0.)})))
    cost=np.hstack((cost,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='float64',delimiter='\t',encoding='latin1',usecols=7,skip_header=1,converters={7: lambda s: float(s or 0.)})))
    status=np.hstack((status,genfromtxt('skucodedetails'+str(i)+'.txt',dtype='object',delimiter='\t',encoding='latin1',usecols=8,skip_header=1,converters={8: lambda s: str(s or '')})))

sku=np.unique(sku,return_inverse=1)
status=np.unique(status,return_inverse=1)

os.chdir('/home/dmf/python/npz/')
np.savez_compressed('ll.npz',sku=sku[1], oid=oid, price=price, units=units, pah=pah, tax=tax, cost=cost,status=status[1],skustr=sku[0],statusstr=status[0])

