import numpy as np
from numpy import genfromtxt
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

arrays=np.load('/home/dmf/python/npz/oldechilo.npz',allow_pickle=1)
olnames=arrays.files
print(olnames)

for x in olnames:
    vars()[x]=np.array(arrays[x])

np.array_equal(oidol,oidol[np.argsort(oidol)])
oidol.size==np.unique(oidol).size
#ORDER_NO	CM_ID	HH_ID	CHUR_TAX_CID	CHUR_TAX_HID

#oida, cida, datea, hida, churchcida, churchhida, skua, pricea, unitsa  = genfromtxt ('/home/dmf/csv/skucodedetails/skucodedetailshigh.txt', dtype= ['int64', 'int64', 'int64', 'int64',  'int64', 'int64', 'int64', 'float64', 'int64'],skip_header = 1, delimiter = '\t', encoding = 'latin1', converters = {0: lambda s: int(s or 0), 1: lambda s: int(s or 0), 2: lambda s: int(s or 0), 3: lambda s: int(s or 0), 4: lambda s: int(s or 0), 5: lambda s: int(s or 0), 6: lambda x: np.searchsorted(parts[:,0],x),  7: lambda s:float(s or 0.), 8:lambda s: int(s or 0)}, unpack=1)

#order_no,cm_id,hh_id,chur_tax_cid,chur_tax_hid = np.genfromtxt('/home/dmf/csv/id2015.txt',delimiter = '\t',skip_header = 1,encoding = 'latin1',unpack = True,dtype='int64')
order_no,cm_id,hh_id = np.genfromtxt('/home/dmf/csv/id2015.txt',delimiter = '\t',skip_header = 1,encoding = 'latin1',unpack = True,usecols = [0,1,2],dtype='int64')
order_no.size==np.unique(order_no).size

i=np.searchsorted(oidol,order_no)
np.where(cid[i] !=cm_id)[0].size
np.where(hid[i] !=hh_id)[0].size
np.where(churchcid[i] !=chur_tax_cid)[0].size
np.where(churchhid[i] !=chur_tax_hid)[0].size

cid[i]=cm_id
hid[i]=hh_id
#churchcid[i]=chur_tax_cid
#churchhid[i]=chur_tax_hid

np.savez_compressed('oldechilo.npz',oidol=oidol, cid=cid, date=date, hid=hid, churchcid=churchcid, churchhid=churchhid, channel=channel,channelkey=channelkey, pay=pay,paykey=paykey, zipcode=zipcode,zipkey=zipkey, state=state,statekey=statekey, new=new, entry=entry,entrykey=entrykey, gender=gender,genderkey=genderkey, member=member,memberkey=memberkey)


