import numpy as np
import gc
import os
from numpy import char
from numpy import genfromtxt

start_year=1997
end_year=2021
num_years=end_year-start_year+1

np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

gc.enable()

#LOAD FILES
os.chdir('/home/dmf/python/npz')

for i in range (start_year,end_year+1):
    vars()['dict'+str(i)]=list()    

for i in range (start_year,end_year+1):
    for j in ['skus_dict','dates_dict','channel_dict','zipcode_dict','new_dict','pay_dict','status_dict']:
        arrays=np.load('arr'+ str(i) + '.npz',allow_pickle=1)
         
        vars()['ol'+str(i)]=np.array((arrays['cid'][0],arrays['cid'][1],arrays['dates'][1],arrays['channel'][1],arrays['zipcode'][1],arrays['new'][1],arrays['pay'][1],arrays['hid'][1]))
        vars()['sl'+str(i)]=np.array((arrays['oid'],arrays['skus'],arrays['price'],arrays['cost'],arrays['units'],arrays['pah'],arrays['tax'],arrays['status']))
        
        vars()['dict'+str(i)].append(arrays[j])
        
        vars()['ol'+str(i)]=np.delete(vars()['ol'+str(i)],np.where( (vars()['ol'+str(i)][0]<1000000) | (vars()['ol'+str(i)][0]==12345678) ),axis=1)
        
        vars()['sl'+str(i)]=np.delete(vars()['sl'+str(i)],np.where( (vars()['sl'+str(i)][0]<1000000) | (vars()['sl'+str(i)][0]==12345678) ),axis=1)

x=np.array(dir())
x=x[np.where((char.startswith(x,'dict'))|(char.startswith(x,'ol'))|(char.startswith(x,'sl'))) ]
print(x)

#ADD PARTS FIELDS TO SKU DICTIONARY
os.chdir('/home/dmf/csv')
parts=genfromtxt('parts.txt',skip_header=1,delimiter=',',dtype='str',encoding='latin1',usecols=(6,11,12),converters={6: lambda s: str(s or ''),11: lambda s: str(s or ''),12: lambda s: str(s or '')})
parts=parts[parts[:,0].argsort()]
tiers=char.partition(parts[:,1],'-')
tier1=tiers[:,0]
tiers=char.partition(tiers[:,2],'-')
tier2=tiers[:,0]
tier3=tiers[:,2]
del tiers
parts=np.vstack((parts.T,tier1,tier2,tier3))

for i in range(start_year,end_year+1):
    partslookup=np.searchsorted(parts[0],vars()['dict'+str(i)][0][:,1])
    cat=parts[1,][partslookup]
    bg=parts[2,][partslookup]
    t1=parts[3,][partslookup]
    t2=parts[4,][partslookup]
    t3=parts[5,][partslookup]
    vars()['dict'+str(i)][0]=np.hstack((vars()['dict'+str(i)][0],cat.reshape(-1,1),bg.reshape(-1,1),t1.reshape(-1,1),t2.reshape(-1,1),t3.reshape(-1,1)))


#STACK OL AND SL ARRAYS
ol=np.empty((8,0),dtype=int)
sl=np.empty((8,0))
dct=[]
for i in range(start_year,end_year+1):
    ol=np.hstack((ol,vars()['ol'+str(i)]))
    sl=np.hstack((sl,vars()['sl'+str(i)]))
    dct.append((vars()['dict'+str(i)]))

olidx=np.zeros((2,num_years),dtype=int)
slidx=np.zeros((2,num_years),dtype=int)
i=0

for j in range (start_year,end_year+1):
    olidx[:1,i]=j
    olidx[1:2,i]=vars()['ol'+str(j)].shape[1]
    slidx[:1,i]=j
    slidx[1:2,i]=vars()['sl'+str(j)].shape[1]
    i=i+1

#save compressed file of all orders
os.chdir('/home/dmf/python/npz')
np.savez_compressed('ordersall.npz',olidx=olidx,slidx=slidx,ol=ol,sl=sl)

#pickle dictionaries
import pickle
with open ('dct.txt','wb') as fp:
    pickle.dump(dct,fp)

