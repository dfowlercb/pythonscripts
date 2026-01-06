import pandas as pd
import numpy as np
import subprocess
from numpy.lib import recfunctions as rfn
from scipy import stats as st
from scipy import sparse
np.set_printoptions(suppress=True)

#USER INPUT VARIABLES
startdate=np.datetime64('2024-07-01').astype(int)
enddate=np.datetime64('2024-07-31').astype(int)

#CLEAN AND LOAD FILES
subprocess.run(["sed", "-i", "'s/[#$\"%!]//g'","~/csv/SKU_OFFICERCAT_RPT.txt"])
subprocess.run(["sed", "-i", "'s/[#$\"%!=]//g'","~/csv/SKU_AVG_SHP_07_28_2024.csv"])
subprocess.run(["sed", "-i", "'s/[#$\"%!=]//g'","~/csv/t2defs.txt"])

arr_0=np.load('/home/dfowler/npz/l24.npz')['arr_0']

cb=pd.read_csv('/home/dfowler/csv/COMPETE/COMP_PRICES_072024_CB.txt',delimiter='\t',encoding='latin1')

comp=pd.read_csv('/home/dfowler/csv/COMPETE/COMP_PRICES_072024_COMP.txt',delimiter='\t',encoding='latin1')

parts=np.loadtxt('/home/dfowler/csv/parts.txt',dtype=[('part' ,'U13'),('DESC' ,'U24'),('PUB' ,'U3'), ('cat' ,'U9')],delimiter=',',encoding='latin1',usecols=(6,7,10,11),skiprows=1)

oc=np.loadtxt('/home/dfowler/csv/SKU_OFFICERCAT_RPT.txt',dtype=[('SKU' ,'U13' ), ('OC' ,'U20' )],delimiter='\t',usecols=(0,1),encoding='latin1',skiprows=1)

t2defs=np.loadtxt('/home/dfowler/csv/t2defs.txt',dtype=[('CODE' ,'U3' ), ('DESC' ,'U24' )],delimiter='\t',usecols=(1,2),encoding='latin1',skiprows=1)

skuavg=np.loadtxt('/home/dfowler/csv/SKU_AVG_SHP_07_28_2024.csv',dtype=[('SKU' ,'U13'), ('COST' ,'f8'), ('UNITS' ,'f8' )],delimiter=',',usecols=(0,4,5),encoding='latin1',skiprows=1)

#MODIFY arr_0
idx=np.where(np.logical_and(arr_0['ORDER_DATE']>=startdate, arr_0['ORDER_DATE']<=enddate))
arr_0=arr_0[idx]

#AGGREGATE arr_0
def agg(n):
    row=np.unique(arr_0['SKU'],return_inverse=True)[1]
    col=np.repeat(0,len(row))
    coo=sparse.coo_matrix((n,(row,col))).toarray()
    return coo

sales=agg(arr_0['UNITS']*arr_0['UNIT_PRICE'])
units=agg(arr_0['UNITS'])
cogs=agg(arr_0['UNITS']*arr_0['UNIT_COST'])
margin=sales-cogs
pah=agg(arr_0['PAH'])

#MODE CALC
arr_cb_data=cb.iloc[:,3:].to_numpy(dtype='float', na_value=np.nan)
arr_comp_data=comp.iloc[:,3:].to_numpy(dtype='float', na_value=np.nan)
cb_mode=np.apply_along_axis(st.mode,1,arr_cb_data,nan_policy='omit')[:,0]
comp_mode=np.apply_along_axis(st.mode,1,arr_comp_data,nan_policy='omit')[:,0]

arr_cb_string=cb.iloc[:,:1].to_numpy().astype('U13')
arr_comp_string=comp.iloc[:,:1].to_numpy().astype('U13')
arr_cb_string=arr_cb_string.reshape(arr_cb_string.size,1)
arr_comp_string=arr_comp_string.reshape(arr_comp_string.size,1)
dt=np.dtype([ ('SKU','U13')])
modelookup=rfn.unstructured_to_structured(arr_cb_string,dtype=dt)
modelookup=rfn.append_fields(modelookup,('AMZN_PRICE','CB_PRICE'),(comp_mode,cb_mode),usemask=False)

idx=np.isin(modelookup['SKU'],arr_0['SKU'])
modelookup=modelookup[idx]

#BUILD EXPORT
export=np.unique(arr_0['SKU'])
export=np.reshape(export,(export.size,1))
dt=np.dtype([ ('SKU','U13')])
export=rfn.unstructured_to_structured(export,dtype=dt)
export=np.sort(export,order='SKU')

parts=np.sort(parts,order='part')
idx=np.searchsorted(parts['part'],export['SKU'])
DESC=parts['DESC'][idx]
PUB=parts['PUB'][idx]
CAT=parts['cat'][idx]
tiers=np.char.split(CAT,sep='-')
t1 = np.array([value[0] for value in tiers])
t2 = np.array([value[1] for value in tiers])
t3 = np.array([value[2] for value in tiers])

t2defs=np.sort(t2defs,order='CODE')
idx=np.searchsorted(t2defs['CODE'],t2)
T2DEF=t2defs['DESC'][idx]

oc=np.sort(oc,order='SKU')
idx=np.searchsorted(oc['SKU'],export['SKU'])
OC=oc['OC'][idx]

idx=np.isin(skuavg['SKU'],arr_0['SKU'])
skuavg=skuavg[idx]
skuavg=np.sort(skuavg,order='SKU')
idx=np.isin(export['SKU'],skuavg['SKU'])
AVG_SHIP=np.full(len(export),np.nan)
AVG_SHIP[idx]=np.round(skuavg['COST']/skuavg['UNITS'],2)

avg01=np.round(np.nanmean(AVG_SHIP[t1=='01']),2)
avg02=np.round(np.nanmean(AVG_SHIP[t1=='02']),2)
avg05=np.round(np.nanmean(AVG_SHIP[t1=='05']),2)
avg07=np.round(np.nanmean(AVG_SHIP[t1=='07']),2)
avg08=np.round(np.nanmean(AVG_SHIP[t1=='08']),2)
avg09=np.round(np.nanmean(AVG_SHIP[t1=='09']),2)
avg10=np.round(np.nanmean(AVG_SHIP[t1=='10']),2)

AVG_SHIP[np.logical_and(np.isnan(AVG_SHIP)==True,t1=='01')]=avg01
AVG_SHIP[np.logical_and(np.isnan(AVG_SHIP)==True,t1=='02')]=avg02
AVG_SHIP[np.logical_and(np.isnan(AVG_SHIP)==True,t1=='05')]=avg05
AVG_SHIP[np.logical_and(np.isnan(AVG_SHIP)==True,t1=='07')]=avg07
AVG_SHIP[np.logical_and(np.isnan(AVG_SHIP)==True,t1=='08')]=avg08
AVG_SHIP[np.logical_and(np.isnan(AVG_SHIP)==True,t1=='09')]=avg09
AVG_SHIP[np.logical_and(np.isnan(AVG_SHIP)==True,t1=='10')]=avg10

AVG_SHIP[np.isin(t3,np.array(['EK','DA','DL','DW','DF']))]=0
AVG_SHIP[np.isin(t1,np.array(['00','04','20']))]=0

modelookup=np.sort(modelookup,order='SKU')
idx=np.isin(export['SKU'],modelookup['SKU'])
AMZN_PRICE=np.full(len(export),np.nan)
CB_PRICE=np.full(len(export),np.nan)
AMZN_PRICE[idx]=modelookup['AMZN_PRICE']
CB_PRICE[idx]=modelookup['CB_PRICE']

#export=rfn.append_fields(export,('DESC','PUB','OC','T2DEF','SALES','UNITS','MARGIN','PAH','AVG_SHIP','AMZN_PRICE','CB_PRICE'),(DESC,PUB,OC,T2DEF,sales,units,margin,pah,AVG_SHIP,AMZN_PRICE,CB_PRICE),usemask=False)

#CONTRIBUTION MARGIN
RETURNS=.02
INBOUND=.0153
PACKAGING=.025876
SHIP_INCOME=.215094
OUTBOUND=.134232
DC=.030189
CALLCENTER=.014468
CREDITCARD=.029

contribution=(np.round(arr_0['UNIT_PRICE']-(arr_0['UNIT_PRICE']*RETURNS)-arr_0['UNIT_COST']-(arr_0['UNIT_PRICE']*INBOUND)-(arr_0['UNIT_PRICE']*PACKAGING)+(arr_0['UNIT_PRICE']*SHIP_INCOME)-(arr_0['UNIT_PRICE']*OUTBOUND)-(arr_0['UNIT_PRICE']*DC)-(arr_0['UNIT_PRICE']*CALLCENTER)-(arr_0['UNIT_PRICE']*CREDITCARD),2))*arr_0['UNITS']
returns=(np.round((arr_0['UNIT_PRICE']*RETURNS*-1),2))*arr_0['UNITS']
packaging=(np.round((arr_0['UNIT_PRICE']*PACKAGING*-1),2))*arr_0['UNITS']
inbound=(np.round((arr_0['UNIT_PRICE']*INBOUND*-1),2))*arr_0['UNITS']
netshipping=(np.round((arr_0['UNIT_PRICE']*SHIP_INCOME)-(arr_0['UNIT_PRICE']*OUTBOUND),2))*arr_0['UNITS']
fulfillment=(np.round(((arr_0['UNIT_PRICE']*DC)+(arr_0['UNIT_PRICE']*CALLCENTER))*-1,2))*arr_0['UNITS']
ccfee=(np.round(arr_0['UNIT_PRICE']*CREDITCARD*-1,2))*arr_0['UNITS']

CONTRIBUTION1=agg(contribution)
RETURNS1=agg(returns)
PACKAGING1=agg(packaging)
INBOUND1=agg(inbound)
NETSHIPPING1=agg(netshipping)
FULFILLMENT1=agg(fulfillment)
CCFEE1=agg(ccfee)

export=rfn.append_fields(export,('DESC','PUB','OC','T2DEF','SALES','UNITS','MARGIN','PAH','CONTRIBUTION','NETSHIPPING','FULFILLMENT','CCFEE','RETURNS','PACKAGING','INBOUND','AVG_SHIP','AMZN_PRICE','CB_PRICE'),(DESC,PUB,OC,T2DEF,sales,units,margin,pah,CONTRIBUTION1,NETSHIPPING1,FULFILLMENT1,CCFEE1,RETURNS1,PACKAGING1,INBOUND1,AVG_SHIP,AMZN_PRICE,CB_PRICE),usemask=False)

export=export[export['SALES']>0]

np.savetxt('/home/dfowler/documents/compete.txt',export,delimiter=',',fmt='%s,%s,%s,%s,%s,%10.2f,%i,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f',header='SKU,DESC,PUB,OC,T2DEF,SALES,UNITS,MARGIN,PAH,CONTRIBUTION,NETSHIPPING,FULFILLMENT,CCFEE,RETURNS,PACKAGING,INBOUND,AVG_SHIP,AMZN_PRICE,CB_PRICE',comments='')

subprocess.run(["rclone", "copy", "/home/dfowler/documents/compete.txt","onedrive:COMPETE/"])


