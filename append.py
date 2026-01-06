import numpy as np
import os
from numpy import genfromtxt
from numpy import char

np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

#%% import existing arrays
os.chdir('/home/dmf/python/npz')

arrays = np.load ('olsephilo.npz')
olnames=arrays.files
olnames = [e for e in olnames if e not in ('channelkey','paykey','zipkey','statekey','entrykey','genderkey','memberkey')]
print(olnames)

for x in olnames:
    vars()[x]=np.array(arrays[x])

arrays = np.load ('llsephilo.npz')
llnames=arrays.files
llnames = [e for e in llnames if e not in ('oldskukey','statuskey','ll_link')]
print(llnames)

for x in llnames:
    vars()[x]=np.array(arrays[x])

skustr=parts[:,0]
llnames.remove('parts')

#%% DELETE ARRAY RANGES TO BE IMPORTED
i=np.where( (date<np.datetime64('2017-01-01').astype(int)) )[0]

for x in olnames:
    print(x,':',vars()[x].size)
    vars()[x]=vars()[x][i]
    print(x,':',vars()[x].size)

i=np.isin(oid,oidol)

for x in llnames:
    print(x,':',vars()[x].size)
    vars()[x]=vars()[x][i]
    print(x,':',vars()[x].size)

#%% IMPORT LATEST PARTS FILE AND RECODE EXISTING SKUS
parts=genfromtxt('/home/dmf/csv/parts.txt',skip_header=1,delimiter=',',dtype='str',encoding='latin1',usecols=(6,11,12),converters={6: lambda s: str(s or ''),11: lambda s: str(s or ''),12: lambda s: str(s or '')})
parts=parts[char.str_len(parts[:,1])==9]
parts=parts[parts[:,0].argsort(),]
tiers=np.concatenate(char.rsplit(parts[:,1],'-'))
t1=np.copy(tiers[::3]).reshape(-1,1)
t2=np.copy(tiers[1::3]).reshape(-1,1)
t3=np.copy(tiers[2::3]).reshape(-1,1)
print(np.unique(t1))
print(np.unique(t2))
print(np.unique(t3))
parts=np.hstack((parts,t1,t2,t3))

for i in np.arange(1,6):
    parts=np.hstack((parts,np.unique(parts[:,i],return_inverse=1)[1].reshape(-1,1)))

i=np.searchsorted(parts[:,0],skustr)
sku=i[sku]
kitsku=sku

# IMPORT MAINFRAME ARRAYS UPDATE
 #ORDER LEVEL
#%% string to number keys
statekey = np.unique(np.array(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY','NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR', 'DC','AE','VI','AP','GU']))

entrykey=np.unique(np.array(['CBD', 'CHRISTIANBOOK.COM', 'MOBILE CHRISTIANBOOK.COM']))

genderkey=np.unique(np.array(['M','F','B','U']))

memberkey=np.unique(np.array(['', 'A', 'B', 'C', 'D', 'W', 'X', 'Y', 'Z']))

zipkey=genfromtxt('/home/dmf/csv/zipcountymsa.csv',skip_header=1,delimiter=',',usecols=0,dtype='str')
zipkey=np.append(zipkey,['INTL','CAND'])
zipkey=np.unique(zipkey)

 #%%genfromtxt high file
#ORDER_NO	0
#CM_ID	1
#ORDER_DATE	2
#HH_ID	3
#CHUR_TAX_CID	4
#CHUR_TAX_HID	5
#SKU	6
#KIT SKU	7
#UNIT_PRICE	8
#UNITS	9

os.chdir('/home/dmf/csv/skucodedetails')
oidola=np.genfromtxt('scdhigh.txt',delimiter='\t',skip_header=1,usecols=(0),dtype='int64')
cida=np.genfromtxt('scdhigh.txt',delimiter='\t',skip_header=1,usecols=(1),dtype='int64')
datea=np.genfromtxt('scdhigh.txt',delimiter='\t',skip_header=1,usecols=(2),dtype='int64')
hida=np.genfromtxt('scdhigh.txt',delimiter='\t',skip_header=1,usecols=(3),dtype='int64')
churchcida=np.genfromtxt('scdhigh.txt',delimiter='\t',skip_header=1,usecols=(4),dtype='int64')
churchhida=np.genfromtxt('scdhigh.txt',delimiter='\t',skip_header=1,usecols=(5),dtype='int64')

churchcida[np.where(churchcida==-1)]=0
churchcida[np.where(churchcida>0)]=1
churchhida[np.where(churchhida==-1)]=0
churchhida[np.where(churchhida>0)]=1

 #dedupe
a_names=tuple(('oidola','cida','datea','hida','churchcida','churchhida'))
i=np.unique(oidola,return_index=1)[1]
for x in a_names:
    vars()[x]=vars()[x][i]

#genfromtxt low file
#ORDER_NO	0
#ORDER_TYPE	1
#FORM_PAY	2
#BILLTO_ZIPCODE	3
#BILLTO_STATE	4
#NEW_CUST_FLAG	5
#ENTRY_METHD	6
#GENDER	7
#MEMBER_CODE	8
#SKU	9
#KIT SKU	10
#UNIT_COST	11
#PAH	12
#TAX	13
#LINE STATUS	14
#LINE_SHIPDATE	15

oidoltemp = genfromtxt ('scdlow.txt', dtype='int64', delimiter='\t', encoding='latin1', usecols=0, skip_header=1, converters = {0: lambda s: int(s or 0)})
channela = genfromtxt ('scdlow.txt', dtype='int64', delimiter='\t', encoding='latin1', usecols=1, skip_header=1, converters = {1:lambda x:{'':0,'FB':1,'FI':2,'FM':3,'FN':4,'FP':5,'FW':6,'I':7,'M':8,'P':9,'S':10,'T':11,'W':12,'X':13}.get(x,0),})

paya = genfromtxt ('scdlow.txt', dtype='int64', delimiter='\t', encoding='latin1', usecols=2, skip_header=1, converters = {2:lambda x:{'AX':7,'CK':9,'CR':10,'CS':11,'DS':13,'EB':14,'EC':15,'FF':17,'GC':18,'MC':20,'NC':22,'PP':23,'TD':24,'TM':25,'TV':26,'VA':27}.get(x,0)})

zipcodea = genfromtxt ('scdlow.txt', dtype='str', delimiter='\t', encoding='latin1', usecols=3, skip_header=1, converters = {1: lambda x: str(x or '')})
zipcodea[np.where(char.str_len(zipcodea)<5)]='INTL'
zipcodea[np.where(zipcodea=='99999')]='INTL'
zipcodea[np.where(char.str_len(zipcodea)>=6)]='CAND'
zipcodea=np.searchsorted(zipkey,zipcodea)

statea = genfromtxt ('scdlow.txt', dtype='str', delimiter='\t', encoding='latin1', usecols=4, skip_header=1, converters = {2: lambda s: str(s or '')})
statea[np.isin(statea,statekey,invert=1)]='zz'
statea=np.searchsorted(statekey,statea,side='right')

newa = genfromtxt ('scdlow.txt', dtype='int64', delimiter='\t', encoding='latin1', usecols=5, skip_header=1)

entrya = genfromtxt ('scdlow.txt', dtype='str', delimiter='\t', encoding='latin1', usecols=6, skip_header=1, converters = {3: lambda s: str(s or '')})
entrya[np.isin(entrya,entrykey,invert=1)]='CBD'
entrya=np.searchsorted(entrykey,entrya)

gendera = genfromtxt ('scdlow.txt', dtype='str', delimiter='\t', encoding='latin1', usecols=7, skip_header=1, converters = {4: lambda s: str(s or '')})
gendera[np.isin(gendera,genderkey,invert=1)]='U'
gendera=np.searchsorted(genderkey,gendera)

membera = genfromtxt ('scdlow.txt', dtype='str', delimiter='\t', encoding='latin1', usecols=8, skip_header=1, converters = {5: lambda s: str(s or '')})
membera[np.isin(membera,memberkey,invert=1)]=''
membera=np.searchsorted(memberkey,membera)

 #dedupe
a_names=tuple(('oidoltemp','channela','paya','zipcodea','statea','newa','entrya','gendera','membera'))
i=np.unique(oidoltemp,return_index=1)[1]
for x in a_names:
    vars()[x]=vars()[x][i]

#equality test
if np.array_equal(oidola,oidoltemp)==False:
    X=np.intersect1d(oidola,oidoltemp)
    i=np.isin(oidola,X)
    for x in ['oidola','cida','datea','hida','churchcida','churchhida']:
        print(vars()[x].size)
        vars()[x]=vars()[x][i]
        print(vars()[x].size)
    
    i=np.isin(oidoltemp,X)
    for x in ['oidoltemp','channela','paya','zipcodea','statea','newa','entrya','gendera','membera']:
        print(vars()[x].size)
        vars()[x]=vars()[x][i]
        print(vars()[x].size)

np.array_equal(oidola,oidoltemp)

#stack appended data
oidol=np.hstack((oidol,oidola))
cid=np.hstack((cid,cida))
date=np.hstack((date,datea))
hid=np.hstack((hid,hida))
churchcid=np.hstack((churchcid,churchcida))
churchhid=np.hstack((churchhid,churchhida))
channel=np.hstack((channel,channela))
pay=np.hstack((pay,paya))
zipcode=np.hstack((zipcode,zipcodea))
state=np.hstack((state,statea))
new=np.hstack((new,newa))
entry=np.hstack((entry,entrya))
gender=np.hstack((gender,gendera))
member=np.hstack((member,membera))

#sort order level
ol_tup=tuple(('oidol','cid','date','hid','churchcid','churchhid','channel','pay','zipcode','state','new','entry','gender','member'))
i=np.argsort(oidol)
for x in ol_tup:
    vars()[x]=vars()[x][i]

#LINE LEVEL
#ORDER_NO	0
#CM_ID	1
#ORDER_DATE	2
#HH_ID	3
#CHUR_TAX_CID	4
#CHUR_TAX_HID	5
#SKU	6
#KIT SKU	7
#UNIT_PRICE	8
#UNITS	9

oida=np.genfromtxt('scdhigh.txt',delimiter='\t',skip_header=1,usecols=(0),dtype='int64')
skua=np.genfromtxt('scdhigh.txt',delimiter='\t',skip_header=1,usecols=(6),dtype='int64',converters={6: lambda x: np.searchsorted(parts[:,0],x)})
kitskua=np.genfromtxt('scdhigh.txt',delimiter='\t',skip_header=1,usecols=(7),dtype='int64',converters={7: lambda x: np.searchsorted(parts[:,0],x)})
pricea=np.genfromtxt('scdhigh.txt',delimiter='\t',skip_header=1,usecols=(8),dtype='float',converters={8: lambda x: float(x or 0.)})
unitsa=np.genfromtxt('scdhigh.txt',delimiter='\t',skip_header=1,usecols=(9),dtype='int64',converters={9: lambda x: int(x or 0)})

#ORDER_NO	0
#ORDER_TYPE	1
#FORM_PAY	2
#BILLTO_ZIPCODE	3
#BILLTO_STATE	4
#NEW_CUST_FLAG	5
#ENTRY_METHD	6
#GENDER	7
#MEMBER_CODE	8
#SKU	9
#KIT SKU	10
#UNIT_COST	11
#PAH	12
#TAX	13
#LINE STATUS	14
#LINE_SHIPDATE	15

oidatemp=np.genfromtxt('scdlow.txt',delimiter='\t',skip_header=1,usecols=(0),dtype='int64')
skutemp=np.genfromtxt('scdlow.txt',delimiter='\t',skip_header=1,usecols=(9),dtype='int64',converters={9: lambda x: np.searchsorted(parts[:,0],x)})
kitskutemp=np.genfromtxt('scdlow.txt',delimiter='\t',skip_header=1,usecols=(10),dtype='int64',converters={10: lambda x: np.searchsorted(parts[:,0],x)})
costa=np.genfromtxt('scdlow.txt',delimiter='\t',skip_header=1,usecols=(11),dtype='float',converters={11: lambda x: float(x or 0.)})
paha=np.genfromtxt('scdlow.txt',delimiter='\t',skip_header=1,usecols=(12),dtype='float',converters={12: lambda x: float(x or 0.)})
taxa=np.genfromtxt('scdlow.txt',delimiter='\t',skip_header=1,usecols=(13),dtype='float',converters={13: lambda x: float(x or 0.)})
statusa=np.genfromtxt('scdlow.txt',delimiter='\t',skip_header=1,usecols=(14),dtype='int64')
shippeda=np.genfromtxt('scdlow.txt',delimiter='\t',skip_header=1,usecols=(15),dtype='int64')

#both files reduced to shared order #s
if np.setdiff1d(oida,oidatemp).size>0 or np.setdiff1d(oidatemp,oida).size>0:
    X=np.intersect1d(oida,oidatemp)
    i=np.isin(oida,X)
    for x in ['oida','skua','kitskua','pricea','unitsa']:
        print(vars()[x].size)
        vars()[x]=vars()[x][i]
        print(vars()[x].size)
    
    i=np.isin(oidatemp,X)
    for x in ['oidatemp','skutemp','kitskutemp','costa','paha','taxa','statusa','shippeda']:
        print(vars()[x].size)
        vars()[x]=vars()[x][i]
        print(vars()[x].size)

#both files reduced to same number of order lines
a=np.unique(oida,return_counts=1)
b=np.unique(oidatemp,return_counts=1)
diffnolines=a[0][np.where(a[1]!=b[1])]

i=np.isin(oida,diffnolines,invert=1)
for x in ['oida','skua','kitskua','pricea','unitsa']:
    print(vars()[x].size)
    vars()[x]=vars()[x][i]
    print(vars()[x].size)

i=np.isin(oidatemp,diffnolines,invert=1)
for x in ['oidatemp','skutemp','kitskutemp','costa','paha','taxa','statusa','shippeda']:
    print(vars()[x].size)
    vars()[x]=vars()[x][i]
    print(vars()[x].size)

#SORT FILES BY ORDER# AND SKU#
i=np.lexsort((skua,oida))
for x in ['oida','skua','kitskua','pricea','unitsa']:
    vars()[x]=vars()[x][i]

i=np.lexsort((skutemp,oidatemp))
for x in ['oidatemp','skutemp','kitskutemp','costa','paha','taxa','statusa','shippeda']:
    vars()[x]=vars()[x][i]

####CRITICAL CHECK#####
np.array_equal(oida,oidatemp)
np.array_equal(skua,skutemp)
####CRITICAL CHECK#####


shipped=np.hstack((np.full(oid.size,-1),shippeda))

oid=np.hstack((oid,oida))
kitsku=np.hstack((kitsku,kitskua))
sku=np.hstack((sku,skua))
units=np.hstack((units,unitsa))
price=np.hstack((price,pricea))
pah=np.hstack((pah,paha))
tax=np.hstack((tax,taxa))
cost=np.hstack((cost,costa))
status=np.hstack((status,statusa))

#sort line level
ll_tup=tuple(('oid','sku', 'kitsku', 'units', 'price', 'pah', 'tax', 'cost', 'status', 'shipped'))
i=np.lexsort((sku,oid))
for x in ll_tup:
    vars()[x]=vars()[x][i]

ll_link=np.searchsorted(oidol,oid)

os.chdir('/home/dmf/python/npz/')
np.savez_compressed('olocthilo.npz',oidol=oidol, cid=cid, date=date, hid=hid, churchcid=churchcid, churchhid=churchhid, channel=channel,pay=pay,zipcode=zipcode,zipkey=zipkey, state=state,statekey=statekey, new=new, entry=entry,entrykey=entrykey, gender=gender,genderkey=genderkey, member=member,memberkey=memberkey)
np.savez_compressed('llocthilo.npz',oid=oid,sku=sku,kitsku=kitsku,units=units,price=price,pah=pah,tax=tax,cost=cost,status=status,shipped=shipped,parts=parts,ll_link=ll_link)







