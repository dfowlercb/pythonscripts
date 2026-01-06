import numpy as np
from numpy.lib import recfunctions as rfn

y22=np.load('/home/dfowler/npz/l22.npz')['arr_0']
y23=np.load('/home/dfowler/npz/l23.npz')['arr_0']
orders=rfn.stack_arrays((y22,y23),usemask=False)
idx=orders['LINE_SHIPDATE']-orders['ORDER_DATE']
orders=orders[np.where(idx>=2)]
DATE=orders['ORDER_DATE'].astype('datetime64[D]')
SHIPDATE=orders['LINE_SHIPDATE'].astype('datetime64[D]')
orders=rfn.append_fields(orders,('DATE','SHIPDATE'),(DATE,SHIPDATE),usemask=False)

reship=orders[orders['SKU']=='RESHIP']['ORDER_NO']
orders=orders[np.isin(orders['ORDER_NO'],reship,invert=True)]

x=np.array(orders.dtype.names)
x=x[np.isin(x,('ORDER_NO','DATE','SHIPDATE','SKU','UNIT_COST','UNITS','UNIT_PRICE','PAH'),invert=True)]
orders=rfn.drop_fields(orders,x)

parts=np.loadtxt('/home/dfowler/csv/parts.txt',usecols=(6,7,10,11,48),delimiter=',',encoding='latin1',skiprows=1,dtype=[('SKU','U21'),('DESC','U50'),('PUB','U4'),('CAT','U9'),('DATEPUB','U10')])
parts=np.sort(parts,order='SKU')
category=np.loadtxt('/home/dfowler/csv/SKU_OFFICERCAT_RPT.txt',delimiter='\t',encoding='latin1',skiprows=1,dtype=[('SKU','U21'),('CAT','U50')])
category=np.sort(category,order='SKU')
classification=np.loadtxt('/home/dfowler/csv/SKUMEDIA.csv',delimiter=',',encoding='latin1',skiprows=1,dtype=[('SKU','U21'),('INDIV','f8'),('COMBO','f8'),('CLASS','U12')])
classification=np.sort(classification,order='SKU')

idx=np.searchsorted(parts['SKU'],orders['SKU'])
desc=parts['DESC'][idx]
pub=parts['PUB'][idx]
datepub=parts['DATEPUB'][idx]
idx=np.searchsorted(category['SKU'],orders['SKU'])
cat=category['CAT'][idx]
idx=np.searchsorted(classification['SKU'],orders['SKU'])
indiv=classification['INDIV'][idx]
combo=classification['COMBO'][idx]
media=classification['CLASS'][idx]

orders=rfn.append_fields(orders,('DESC','PUB','CAT','FORMAT','DATEPUB','INDIV','COMBO'),(desc,pub,cat,media,datepub,indiv,combo),usemask=False)

def findday(arr,day):
    y=np.is_busday(arr, weekmask=day)
    return(y)

findday_v = np.vectorize(findday)

friday=findday_v(orders['DATE'],'Fri')
interval=orders['SHIPDATE']-orders['DATE']
interval=interval.astype(int)
idx1=np.where(friday==True)
idx2=np.where(interval==3)
idx3=np.intersect1d(idx1[0],idx2[0])
exclude=orders['ORDER_NO'][idx3]
orders=orders[np.isin(orders['ORDER_NO'],exclude,invert=True)]

saturday=findday_v(orders['DATE'],'Sat')
interval=orders['SHIPDATE']-orders['DATE']
interval=interval.astype(int)
idx1=np.where(saturday==True)
idx2=np.where(interval==2)
idx3=np.intersect1d(idx1[0],idx2[0])
exclude=orders['ORDER_NO'][idx3]
orders=orders[np.isin(orders['ORDER_NO'],exclude,invert=True)]

fn=orders.dtype.names
fn=','.join(map(str,fn))
np.savetxt('/home/dfowler/documents/backorders.txt',orders,delimiter=',',header=fn,comments='',fmt='%i,%s,%1.2f,%1.2f,%1.2f,%i,%s,%s,%s,%s,%s,%s,%s,%1.2f,%1.2f')







