import numpy as np
import numpy.lib.recfunctions as rf
from partsconversion import convparts

olhigh=np.load('/home/dmf/python/npz/olhigh.npz')
date=olhigh['date']
llhigh=np.load('/home/dmf/python/npz/llhigh.npz')
oid=llhigh['oid']
sku=llhigh['sku']
units=llhigh['units']
price=llhigh['price']
link=llhigh['ll_link']
parts=llhigh['parts']

ship=np.genfromtxt('/home/dmf/Downloads/test.txt',delimiter='\t',skip_header=1,usecols=(0,10,15),dtype=[ ('ORDER_NO','i4'),('KITSKU','U13'),('LINE_SHIPDATE','i4') ])
bib=np.genfromtxt('/home/dmf/csv/bibskucodedetails.txt',delimiter='\t',skip_header=1,usecols=(0,6,7,8,21),dtype=[('ORDERNO', 'i4'), ('SKU', 'U13'), ('UNITPRICE', 'f4'), ('UNITS', 'i4'), ('LINESHIPDATE', 'i4')])
i=np.where( (bib['LINESHIPDATE']>=np.datetime64('2022-07-01').astype(int)) & (bib['LINESHIPDATE']<=np.datetime64('2022-07-31').astype(int)) )
bib=bib[i]

date=date[link]
parts=convparts(parts)
shipsku=np.searchsorted(parts['SKU'],ship['KITSKU'])
shiphash=np.char.add(ship['ORDER_NO'].astype(str),shipsku.astype(str))

arrlist=['oid','date','sku','units','price']
i=np.isin(oid,ship['ORDER_NO'])
for x in arrlist:
    vars()[x]=vars()[x][i]

oidskuhash=np.char.add(oid.astype(str),sku.astype(str))
i=np.isin(oidskuhash,shiphash)
for x in arrlist:
    vars()[x]=vars()[x][i]

oidskuhash=np.char.add(oid.astype(str),sku.astype(str))
i=np.isin(oidskuhash,shiphash)

print(np.sum(units*price)+np.sum(bib['UNITS']*bib['UNITPRICE']))


