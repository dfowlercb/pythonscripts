import numpy as np
import numpy.lib.recfunctions as rf
from scipy import sparse
import os

np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

ol_dtype=[('OID', 'i8'), ('CID', 'i8'), ('DATE', 'i8'), ('SKU', 'U13'), ('PRICE', 'f8'), ('UNITS', 'i8')]
ll_dtype=[('OID', 'i8'), ('STATE', 'U2'), ('NEW', 'i8'), ('SKU', 'U13'), ('COST', 'f8'), ('PAH', 'f8'), ('TAX', 'f8')]
parts_dtype=[('SKU', 'U10'), ('CAT', 'U13')]

os.chdir('/home/dmf/csv/skucodedetails/')
ol=np.genfromtxt('skucodedetailshigh.txt',delimiter='\t',usecols=(0,1,2,6,7,8),skip_header=1,dtype=ol_dtype)
ol['PRICE'][np.isnan(ol['PRICE'])==True]=0
ll=np.genfromtxt('skucodedetailslow.txt',delimiter='\t',usecols=(0,4,5,9,10,11,12),skip_header=1,dtype=ll_dtype)
os.chdir('/home/dmf/csv/')
ll['COST'][np.isnan(ll['COST'])==True]=0
parts=np.genfromtxt('parts.txt',delimiter=",",usecols=(6,11),skip_header=1,dtype=parts_dtype,encoding='latin1')
parts=parts[np.isin(parts['SKU'],ol['SKU'])]

ol=np.sort(ol,order=['OID','SKU'])
ll=np.sort(ll,order=['OID','SKU'])
parts=np.sort(parts,order='SKU')
np.array_equal(ol['OID'],ll['OID'])
np.array_equal(ol['SKU'],ll['SKU'])
i=np.searchsorted(parts['SKU'],ol['SKU'])
cat=parts['CAT'][i]
os.chdir('/home/dmf/python/npz')
np.savez_compressed('shipping.npz',ol=ol,ll=ll,cat=cat)

#######################################
import numpy as np
import numpy.lib.recfunctions as rf
import pandas as pd
from scipy import sparse
import os
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})
pd.options.display.float_format = '{:20,.2f}'.format

arrays=np.load('/home/dmf/python/npz/shipping.npz')
ol=arrays['ol']
ll=arrays['ll']
cat=arrays['cat']

#EXCLUSIONS
 #digital
def rmdigital(ol,ll,cat,sub):
    i=np.char.endswith(cat,sub)
    ol=ol[np.invert(i)]
    ll=ll[np.invert(i)]
    cat=cat[np.invert(i)]
    return(ol,ll,cat)

ol,ll,cat=rmdigital(ol,ll,cat,'EK')
ol,ll,cat=rmdigital(ol,ll,cat,'DL')
ol,ll,cat=rmdigital(ol,ll,cat,'VD')
ol,ll,cat=rmdigital(ol,ll,cat,'DA')
ol,ll,cat=rmdigital(ol,ll,cat,'DF')
ol,ll,cat=rmdigital(ol,ll,cat,'DW')

merch=ol['UNITS']*ol['PRICE']
merch[merch<0]=0
cost=ol['UNITS']*ll['COST']
oid=np.unique(ol['OID'])
ordermerch=np.bincount(ol['OID'],weights=merch)
ordermerch=ordermerch[np.unique(ol['OID'])]
ordercost=np.bincount(ol['OID'],weights=cost)
ordercost=ordercost[np.unique(ol['OID'])]
orderpah=np.bincount(ll['OID'],weights=ll['PAH'])
orderpah=orderpah[np.unique(ol['OID'])]
ordercount=np.repeat(1,len(oid))

orderband=np.ceil(ordermerch/5)*5
orderband[orderband<0]=0
orderband[orderband>100]=101
orderband[np.isnan(orderband)]=0
orderband2=np.unique(orderband,return_inverse=True)

date=ol['DATE'].astype('datetime64[D]')
month=date.astype('datetime64[M]')
month=month.astype('datetime64[D]')
month=month[np.unique(ol['OID'],return_index=1)[1]]
month2=np.unique(month,return_inverse=True)

os.chdir('/home/dmf/csv/')
promo_dtype=[('OID', 'i8'), ('TYPE','i8')]
promo=np.genfromtxt('SPECIAL_OFFERS.txt',delimiter='\t',skip_header=24,dtype=promo_dtype)
promokey=np.genfromtxt('SPECIAL_OFFERS.txt',delimiter='\t',max_rows=24,dtype='str')
promo=promo[np.isin(promo['OID'],oid)]
promo=np.sort(promo,order='OID')
promocode=np.repeat(0,len(oid))
i=np.searchsorted(oid,promo['OID'])
promocode[i]=promo['TYPE']


shipcost_dtype=[('OID', 'i8'), ('METH', 'U13'), ('ORDER_PAH', 'f8'), ('SHIP_COST', 'f8')]
shipcost=np.genfromtxt('order_ship_cost.txt',delimiter='\t',skip_header=1,usecols=(0,1,2,3),dtype=shipcost_dtype)
shipcost=shipcost[np.isin(shipcost['OID'],oid)]
shipcost=np.sort(shipcost,order='OID')
method=np.repeat('AAAAAAAA',len(oid))
carrier=np.repeat(0.,len(oid))
pah=np.repeat(0.,len(oid))
i=np.searchsorted(oid,shipcost['OID'])
method[i]=shipcost['METH']
carrier[i]=shipcost['SHIP_COST']
pah[i]=shipcost['ORDER_PAH']

cutoff=np.datetime64('2018-01-01').astype(int)
rollup_dtype=np.dtype([('date' ,'i8' ),('oid' ,'i8' ),('band','i8'), ('merch','f8'),('pahcb','f8'),('pahchg','f8'),('method','U5'),('promo','i8')])
rollup=rf.unstructured_to_structured(np.array([month.astype(int),oid,orderband.astype(int),ordermerch,pah,carrier,method,promocode]).T,dtype=rollup_dtype)
rollup=np.delete(rollup,rollup['date']<cutoff,axis=0)
rollup=np.delete(rollup,rollup['method']=='AAAAA',axis=0)
rollup=rf.append_fields(rollup, 'date2', rollup['date'].astype('datetime64[D]'),usemask=False)

df=pd.DataFrame(rollup)
df['Year']=df['date2'].dt.year
df['Month']=df['date2'].dt.month
df['pahchg']=df['pahchg']*-1
df['lose']=0
df.loc[df['pahcb']+df['pahchg']<0,'lose']=1

table=pd.pivot_table(df,columns='Year',values=['merch','pahcb','pahchg'],aggfunc=np.sum)
table.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')

table=pd.pivot_table(df,index=['method'],columns='Year',values=['pahcb','pahchg'],aggfunc=np.sum)
table.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')

table=pd.pivot_table(df[df['method']=='STD'],index='band',columns=['Year'],values=['pahcb','pahchg'],aggfunc=np.sum)
table.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')

table=pd.pivot_table(df[df['method']=='FXGRN'],index='band',columns=['Year'],values=['pahcb','pahchg'],aggfunc=np.sum)
table.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')

table=pd.pivot_table(df[df['method']=='FX2DA'],index='band',columns=['Year'],values=['pahcb','pahchg'],aggfunc=np.sum)
table.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')

table=pd.pivot_table(df[df['method']=='CAN'],index='band',columns=['Year'],values=['pahcb','pahchg'],aggfunc=np.sum)
table.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')

table=pd.pivot_table(df[df['method']=='FX1DA'],index='band',columns=['Year'],values=['pahcb','pahchg'],aggfunc=np.sum)
table.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')

table=pd.pivot_table(df[df['method']=='INT'],index='band',columns=['Year'],values=['pahcb','pahchg'],aggfunc=np.sum)
table.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')

table=pd.pivot_table(df[df['method']=='PR'],index='band',columns=['Year'],values=['pahcb','pahchg'],aggfunc=np.sum)
table.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')

table=pd.pivot_table(df,index='promo',columns=['Year'],values=['pahcb','pahchg'],aggfunc=np.sum)
table.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')

df['overcharge']=0
df.loc[df['pahcb']+df['pahchg']>.01,'overcharge']=1
table=pd.pivot_table(df,index='band',columns=['overcharge'],values=['oid'],aggfunc='count')
table.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')

weight=np.genfromtxt('order_ship_cost.txt',delimiter='\t',skip_header=1,usecols=np.arange(5,73,3))
oidetc=np.genfromtxt('order_ship_cost.txt',delimiter='\t',skip_header=1,usecols=(0,2,3),dtype=[('oid' ,'i8' ), ('pahcb' ,'f8' ), ('pahchg' ,'f8' )])
weightsum=np.sum(weight,axis=1)

oidetc=rf.append_fields(oidetc, 'weight', weightsum, usemask=False)
df2=pd.DataFrame(oidetc)
df2['weight']=df2['weight'].round().astype(int)
df2['lose']=0
df2.loc[df2['pahchg']>df2['pahcb'],'lose']=1
table=pd.pivot_table(df2,index='weight',columns='lose',values='oid',aggfunc='count')
table.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')

geo=ll[['OID','STATE']].copy()
geo=np.unique(geo)
i=np.isin(geo['OID'],df2['oid'])
geo=geo[i]
i=np.isin(df2['oid'],geo['OID'])
df2=df2.iloc[i,:]
geo=np.sort(geo,order='OID')
df2=df2.sort_values('oid')
np.array_equal(geo['OID'],df2['oid'])
df3 = pd.concat([df2, pd.DataFrame(geo['STATE'],columns=['STATE'])], axis=1)
table=pd.pivot_table(df3.loc[df3['weight']<11],index=['STATE','weight'],columns=['lose'],values='oid',aggfunc='count')
table.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')

df2=pd.read_csv('order_ship_cost.txt',delimiter='\t')
df2['count']=1
x=pd.pivot_table(df2.loc[(df2['ORDER_PAH']==5.99) & (df2['ORDER_SHIP_METH']=='STD')],index='ORDER_SHIP_METH',values=['count','ORDER_PAH','SHIP_COST','PP_SHIP_COST','FXSPLW_SHIP_COST','MAIL-INN_SHIP_COST','DRPSHP_SHIP_COST','DHL_SHIP_COST','FXSAV_SHIP_COST','MIBPM_SHIP_COST','OTHER_SHIP_COST','BPM_SHIP_COST','FEDXSP_SHIP_COST','FBPMDC_SHIP_COST','FX1DAY_SHIP_COST','FX1DAYP_SHIP_COST','FX2DAY_SHIP_COST','FXGRND_SHIP_COST','FXHOME_SHIP_COST','UPSSP_SHIP_COST','CAN_SHIP_COST','CPU_SHIP_COST','PR_SHIP_COST','UPS_SHIP_COST','METER_SHIP_COST','FINT_SHIP_COST'],aggfunc=sum)
x.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')

arr1=np.array([oidetc['oid'] , weightsum ]).T
arr2=df[['Year','oid','merch','band','pahcb','pahchg']].to_numpy()
arr1=rf.unstructured_to_structured(arr1,names=['oid','weight'])
arr2=rf.unstructured_to_structured(arr2,names=['Year','oid','merch','band','pahcb','pahchg'])
arr1=arr1[np.isin(arr1['oid'],arr2['oid'])]
arr2=arr2[np.isin(arr2['oid'],arr1['oid'])]
arr1=np.sort(arr1,order='oid')
arr2=np.sort(arr2,order='oid')
np.array_equal(arr1['oid'],arr2['oid'])
arr3=rf.append_fields(arr2,'weight',arr1[['weight']],usemask=False)
df3=pd.DataFrame(rf.structured_to_unstructured(arr3),columns=arr3.dtype.names)
df3['count']=1
df3['weight']=df3['weight'].apply(np.ceil)
df3['net']=df3['pahcb']+df3['pahchg']
x=pd.pivot_table(df3.loc[(df3['pahcb']>0) & (df3['Year']==2021)],index=['weight'],columns='band',values='net',aggfunc=sum)
x.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')
x=pd.pivot_table(df3.loc[(df3['pahcb']==0) & (df3['Year']==2021)],index=['weight'],values=['count' ,'merch','net' ],aggfunc=sum)
x.to_csv('/home/dmf/Documents/shipping.txt',sep='\t')
