#save po tuples file with list0= inserted into the first line
#search/replace any '' with '0.'

import numpy as np
from numpy.lib import recfunctions as rfn
from tuples.podata21 import po21
from tuples.podata22 import po22
from datetime import datetime

def convertdate(x):
    x=np.datetime64(datetime.strptime(x,'%m/%d/%Y').date())
    return x

def loadpo(x):
    dt=np.dtype('U15, U4, U20, U10, U10, U20, U20, U20, U5, U10, U10, U21, U21, U9, U4, U10, U21, U21, f4, f4, f4, U10, U10, U4, f4, U10')
    po=np.array(x[1:],dtype=dt)
    po.dtype.names=('PurchaseOrder','VendorID','VendorName','DateEntered','ExpectedDate','ShipVIA','FOB','Terms','Status','InvoiceNumber','DateOrdered','Sku','Desc','Category','Publisher','Buyer','VendorPart','VendorPartDesc','VendorListPrice','OrderQuantity','ReceivedQuantity','DatePosting','ExpectedDelivery ','DiscountOffRetail','RetailPrice','Complete')
    return(po)

po2021=loadpo(po21)
po2022=loadpo(po22)
po=rfn.stack_arrays((po2021,po2022),usemask=False)
#remove qty greater than 24
#idx=np.where(po['OrderQuantity']<=24)
#po=po[idx]

tiers=np.char.split(po['Category'],sep='-')
t1 = np.array([value[0] for value in tiers])
idx=np.where(t1=='05')
po=po[idx]

date=po['DateOrdered']
date=[convertdate(ele) for ele in date]
po=rfn.append_fields(po,'OrderDate',date,usemask=False)

po=rfn.drop_fields(po,('PurchaseOrder','VendorID','VendorName','DateOrdered','DateEntered','ExpectedDate','ShipVIA','FOB','Terms','Status','InvoiceNumber','Buyer','VendorPart','VendorPartDesc','VendorListPrice','ReceivedQuantity','DatePosting','ExpectedDelivery ','DiscountOffRetail','RetailPrice','Complete'))

#remove duplicates
#idx=np.unique(po['Sku'], return_counts=True)
#duplicates=idx[0][np.where(idx[1]>1)]
#idx=np.isin(po['Sku'],duplicates,invert=True)
#po=po[idx]

parts=np.loadtxt('/home/dfowler/csv/parts.txt',dtype=[('part' ,'U13' ),('first_date_received','M8[D]'), ('sales_units_ever' ,'i4' )],delimiter=',',encoding='latin1',usecols=(6,46,74),converters = {46: lambda s: convertdate(s or '1/1/1999')},skiprows=1)
parts=np.sort(parts,order='part')
idx=np.searchsorted(parts['part'],po['Sku'])
salesunitsever=parts['sales_units_ever'][idx]
firstdatereceived=parts['first_date_received'][idx]
po=rfn.append_fields(po,('first_date_rcvd','sales_units_ever'),(firstdatereceived,salesunitsever),usemask=False)

idx=np.where(po['first_date_rcvd']>=np.datetime64('2021-01-01'))
po=po[idx]

np.savetxt('/home/dfowler/documents/giftposkus.txt',po,fmt='%s, %s, %s, %s, %4.2f, %s, %s, %i')

