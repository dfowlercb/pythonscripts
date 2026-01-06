#12.2.22.30 AVAILABLE ORDER FIELDS: ALL,CM.ID,ORDER.NO,ORDER.DATE,SKU.CODE,SKU,TITLE,UNIT.PRICE,UNITS,GENDER,ORDER.TYPE,MEMBER.CODE,BUSINESS.FLAG,CATALOG.CODE,RENTAL.NO,BILLTO.ZIPCODE,PAH,TAX,EXT.PRICE,NEW.CUST.FLAG,KEYCODE,PROMO.CODE,FLOW.CATALOG,FLOW.NAME,CATEGORY,UNIT.COST,FORM.PAY,BILLTO.STATE,ENTRY.METHOD,HH.ID,LINE.STATUS,CHUR.TAX.CID,CHUR.TAX.HID:

#AVAILABLE PARTS FIELDS: ISBN13, ISBN,UPC,PRODUCT,DESC,WEB.TITLE,AUTHOR,PUBLISHER,CATEGORY,BUYER.GROUP,PRODUCT.TYPE,CLOSEOUT.FLAG,DISC.FLAG,STATUS.DETAIL,DAMAGED.FLAG,CBD.EXCL,SPANISH.FLAG,AFRICAN.AMERICAN,KIDS.FLAG,AGE,FREE.PAH.FLAG,KIT.FLAG,VENDOR.RETAIL,RETAIL,CURRENT.PRICE,ONHAND,ON.ORDER,VALUE,MONTHS.LEFT,FIRST.DATE.RCVD,LAST.DATE.RCVD,PUB.DATE


import numpy as np
import os
import gc
from numpy import genfromtxt
from datetime import datetime 

gc.enable()
str2date = lambda x: datetime.strptime(x, '%m/%d/%Y')

for i in range (2005,2006):
    os.chdir('/home/dmf/csv/')
        
    cid, oid, dates, skus, price, units, channel, zipcode, pah, tax, new, cost, pay, hid, status = genfromtxt ('skucodedetails'+str(i)+'.txt', dtype= ['int64', 'int64', 'M8[D]', 'object', 'float64', 'int64', 'object', 'object', 'float64', 'float64', 'object', 'float64', 'object', 'int64', 'object'],usecols = (np.arange(0,15)),skip_header = 1, delimiter = '\t', encoding = 'latin1', converters = {0: lambda s: int(s or 0), 1: lambda s: int(s or 0), 2: lambda s: str2date(s), 3: lambda s: str(s or ''), 4: lambda s: float(s or 0.), 5: lambda s: int(s or 0), 6:lambda s: str(s or ''), 7:lambda s:str(s or ''), 8: lambda s:float(s or 0.), 9:lambda s:float(s or 0.), 10: lambda s: str(s or 'N'), 11: lambda s: float(s or 0.), 12:lambda s: str(s or ''), 13:lambda s: int(s or 0), 14:lambda s:str(s or '')}, unpack=1)
    
    #cid, oid, dates, skus, price, units, channel, zipcode, pah, tax, new, cost, pay, hid, status = genfromtxt ('skucodedetails'+str(i)+'.txt', dtype= ['int64', 'int64', 'M8[D]', 'object', 'float64', 'int64', 'object', 'object', 'float64', 'float64', 'object', 'float64', 'object', 'int64', 'object'],usecols = (0, 1, 2, 4, 6, 7, 9, 14, 15, 16, 18, 24, 25, 28, 29),skip_header = 1, delimiter = '\t', encoding = 'latin1', converters = {0: lambda s: int(s or 0), 1: lambda s: int(s or 0), 2: lambda s: str2date(s), 4: lambda s: str(s or ''), 6: lambda s: float(s or 0.), 7: lambda s: int(s or 0), 9:lambda s: str(s or ''), 14:lambda s:str(s or ''), 15: lambda s:float(s or 0.), 16:lambda s:float(s or 0.), 18: lambda s: str(s or 'N'), 24: lambda s: float(s or 0.), 25:lambda s: str(s or ''), 28:lambda s: int(s or 0), 29:lambda s:str(s or '')}, unpack=1)
    
    dates_dict=np.unique(dates)
    dates=np.searchsorted(dates_dict,dates)
    dates_dict=dict(enumerate(dates_dict.flatten()))
    dates_dict=np.array(list(dates_dict.items()))
    
    skus_dict=np.unique(skus)
    skus=np.searchsorted(skus_dict,skus)
    skus_dict=dict(enumerate(skus_dict.flatten()))
    skus_dict=np.array(list(skus_dict.items()))
     
    channel_dict=np.unique(channel)
    channel=np.searchsorted(channel_dict,channel)
    channel_dict=dict(enumerate(channel_dict.flatten()))
    channel_dict=np.array(list(channel_dict.items()))
    
    zipcode_dict=np.unique(zipcode)
    zipcode=np.searchsorted(zipcode_dict,zipcode)
    zipcode_dict=dict(enumerate(zipcode_dict.flatten()))
    zipcode_dict=np.array(list(zipcode_dict.items()))
    
    new_dict=np.unique(new)
    new=np.searchsorted(new_dict,new)
    new_dict=dict(enumerate(new_dict.flatten()))
    new_dict=np.array(list(new_dict.items()))
    
    pay_dict=np.unique(pay)
    pay=np.searchsorted(pay_dict,pay)
    pay_dict=dict(enumerate(pay_dict.flatten()))
    pay_dict=np.array(list(pay_dict.items()))
    
    status_dict=np.unique(status)
    status=np.searchsorted(status_dict,status)
    status_dict=dict(enumerate(status_dict.flatten()))
    status_dict=np.array(list(status_dict.items()))
     
    #collapse order level attributes
    
    cid=np.unique(np.array((oid,cid)),axis=1)
    dates=np.unique(np.array((oid,dates)),axis=1)
    channel=np.unique(np.array((oid,channel)),axis=1)
    zipcode=np.unique(np.array((oid,zipcode)),axis=1)
    new=np.unique(np.array((oid,new)),axis=1)
    pay=np.unique(np.array((oid,pay)),axis=1)
    hid=np.unique(np.array((oid,hid)),axis=1)
        
     
    os.chdir('/home/dmf/python/npz/')
    np.savez_compressed('arr'+str(i)+'.npz',cid=cid, oid=oid, dates=dates, skus=skus, price=price, units=units, channel=channel, zipcode=zipcode, pah=pah, tax=tax, new=new, cost=cost, pay=pay, hid=hid, status=status, skus_dict=skus_dict, dates_dict=dates_dict,channel_dict=channel_dict,zipcode_dict=zipcode_dict,new_dict=new_dict,pay_dict=pay_dict,status_dict=status_dict)
     
    del(cid, oid, dates, skus, price, units, channel, zipcode, pah, tax, new, cost, pay, hid, status, skus_dict, dates_dict, channel_dict, zipcode_dict,new_dict,status_dict)
    gc.collect()

