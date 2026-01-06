import numpy as np
import sklearn
from scipy import sparse
from sklearn.preprocessing import scale
from numba import njit
import numpy.lib.recfunctions as rf
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

arrays=np.load('/home/dmf/python/npz/olmayhilo.npz')
for i in tuple(('oidol','hid','date','churchhid')):
    vars()[i]=np.array(arrays[i])

arrays=np.load('/home/dmf/python/npz/llmayhilo.npz')
for i in tuple (('oid','sku','units','price','parts','ll_link')):
    vars()[i]=np.array(arrays[i])

#parts to record array
for i in np.arange(6):
    np.max(np.char.str_len(parts[:,i]))

dt = np.dtype([ ('sku','<U16'), ('cat','<U9'), ('bg','<U17'),  ('t1','<U2'),  ('t2','<U3'),  ('t3','<U2'), ('catnum' , 'i8' ), ('bgnum' , 'i8' ),  ('t1num' , 'i8' ),  ('t2num' , 'i8' ),  ('t3num' , 'i8' ) ])
parts=rf.unstructured_to_structured(parts,dtype=dt)

#merch 
merch=np.bincount(oid,units*price)
merch=merch[oidol]

#pro
t2num=np.unique(parts['t2num'][np.isin(parts['t2'],['APO','BBS','BRF','BST','CHH','CHR','CMM','CSL','CUR','DSS','EVA','GNR','INS','LAN','LEA','PHI','PRE','PSY','THE','VBS'])])
t3num=np.unique(parts['t3num'][np.isin(parts['t3'],['GA','PP','EM','BP','LD','YM','PR'])])
idx1=np.isin( parts['t2num'][sku], t2num, invert=True)  
idx2=np.isin( parts['t3num'][sku], t3num, invert=True)  
idx=np.vstack(( idx1, idx2 ))
idx=np.all(idx,axis=0)
sales=units*price
sales[idx]=0
pro=np.bincount(oid,sales)
pro=pro[oidol]

#buyer groups
def bgcalc(*args):
    bgnum=np.unique(parts['bgnum'][np.isin(parts['bg'],args)])
    idx=np.isin(parts['bgnum'][sku],bgnum,invert=True)
    sales=units*price
    sales[idx]=0
    x=np.bincount(oid,sales)
    x=x[oidol]
    return x

hms=bgcalc('HOMESCHOOLING')
gft=bgcalc('GIFTS','CLOTHING')
fic=bgcalc('FICTION')
kds=bgcalc('KIDS')
bib=bgcalc('BIBLES')
aca=bgcalc('ACADEMIC')
clv=bgcalc('CHRISTIAN LIVING')
chu=bgcalc('CHURCH INTEREST')
mfg=bgcalc('MARRIAGE & FAMILY','MEN & WOMEN')
mus=bgcalc('MUSIC')
sea=bgcalc('SEASONAL')
spa=bgcalc('SPANISH')
vid=bgcalc('VIDEO')

#rfm record array
i=np.where(np.round(merch)>0)
dt=np.dtype([ ('hid', np.int64 ), ('oidol',np.int64 ), ('date',np.int64 ), ('church',np.int64 ),  ('merch' , np.float64 ) , ('hms' , np.float64 ) , ('pro' , np.float64 ) , ('gft' , np.float64 ), ('fic' , np.float64 ), ('kds' , np.float64 ), ('bib' , np.float64 ), ('aca' , np.float64 ),  ('clv' , np.float64 ),  ('chu' , np.float64 ),  ('mfg' , np.float64 ),  ('mus' , np.float64 ),  ('sea' , np.float64 ),  ('spa' , np.float64 ),  ('vid' , np.float64 ),  ('rec' ,np.int64 ), ('freq' , np.int64), ('mon' , np.float64), ('hmsmon' ,np.float64), ('promon' ,np.float64), ('gftmon' ,np.float64),  ('ficmon' ,np.float64), ('kdsmon' ,np.float64), ('bibmon' ,np.float64), ('acamon' ,np.float64), ('clvmon' ,np.float64), ('chumon' ,np.float64), ('mfgmon' ,np.float64), ('musmon' ,np.float64), ('seamon' ,np.float64), ('spamon' ,np.float64), ('vidmon' ,np.float64), ('score',np.int64)])
rfm=rf.unstructured_to_structured ( np.array([ hid[i] , oidol[i] , date[i] , churchhid[i], merch[i], hms[i], pro[i], gft[i], fic[i], kds[i], bib[i], aca[i],  clv[i],  chu[i],  mfg[i],  mus[i],  sea[i],  spa[i],  vid[i],   np.repeat(0,hid[i].size), np.repeat(0,hid[i].size), np.repeat(0,hid[i].size), np.repeat(0,hid[i].size), np.repeat(0,hid[i].size), np.repeat(0,hid[i].size), np.repeat(0,hid[i].size), np.repeat(0,hid[i].size), np.repeat(0,hid[i].size), np.repeat(0,hid[i].size),  np.repeat(0,hid[i].size),  np.repeat(0,hid[i].size),  np.repeat(0,hid[i].size),  np.repeat(0,hid[i].size),  np.repeat(0,hid[i].size),  np.repeat(0,hid[i].size),  np.repeat(0,hid[i].size),  np.repeat(-1,hid[i].size)]).T, dtype=dt )
i=np.lexsort(( rfm['oidol'], rfm['date'], rfm['hid'] ))
rfm=rfm[i]

#rolling rfm calcs

def rfmselect(arr,frqmin,frqmax):
    hidcount=np.unique(rfm['hid'],return_counts=True)
    hida=hidcount[0][np.where( (hidcount[1]>frqmin) & (hidcount[1]<=frqmax) )]
    rfmtemp=rfm.copy()[np.isin(rfm['hid'],hida)]
    return(rfmtemp)

def recroll(arr,frqmax):
    for i in np.arange(frqmax):
        rfm2=np.roll(arr['date'],1)
        arr['rec']=arr['date']-rfm2

@njit(parallel=True)
def frqroll(arr,frqmax):
    for i in np.arange(frqmax):
        hid2=np.roll(arr['hid'],i)
        arr['freq'][arr['hid']==hid2]=i

@njit(parallel=True)
def monroll(arr,frqmax):
        for i in np.arange(1,frqmax):
            idx=np.where(arr['freq']==i)
            rfm2=np.roll(arr,1)
            arr['mon'][idx] = rfm2['merch'][idx]+rfm2['mon'][idx]
            arr['hmsmon'][idx] = rfm2['hms'][idx]+rfm2['hmsmon'][idx]
            arr['promon'][idx] = rfm2['pro'][idx]+rfm2['promon'][idx]
            arr['gftmon'][idx] = rfm2['gft'][idx]+rfm2['gftmon'][idx]
            arr['ficmon'][idx] = rfm2['fic'][idx]+rfm2['ficmon'][idx]
            arr['kdsmon'][idx] = rfm2['kds'][idx]+rfm2['kdsmon'][idx]
            arr['bibmon'][idx] = rfm2['bib'][idx]+rfm2['bibmon'][idx]
            arr['acamon'][idx] = rfm2['aca'][idx]+rfm2['acamon'][idx]
            arr['clvmon'][idx] = rfm2['clv'][idx]+rfm2['clvmon'][idx]
            arr['chumon'][idx] = rfm2['chu'][idx]+rfm2['chumon'][idx]
            arr['mfgmon'][idx] = rfm2['mfg'][idx]+rfm2['mfgmon'][idx]
            arr['musmon'][idx] = rfm2['mus'][idx]+rfm2['musmon'][idx]
            arr['seamon'][idx] = rfm2['sea'][idx]+rfm2['seamon'][idx]
            arr['spamon'][idx] = rfm2['spa'][idx]+rfm2['spamon'][idx]
            arr['vidmon'][idx] = rfm2['vid'][idx]+rfm2['vidmon'][idx]

lst1=[ (0,10), (10,50), (50,100), (100,250), (250,500), (500,1000), (1000,5000), (5000,10000) ]
rfmappend=np.zeros_like(rfm[0])
for i in np.arange(len(lst1)):
    print(lst1[i])
    frqmin=lst1[i][0]
    frqmax=lst1[i][1]
    rfmtemp=rfmselect(rfm,frqmin,frqmax)
    recroll(rfmtemp,frqmax)
    frqroll(rfmtemp,frqmax)
    monroll(rfmtemp,frqmax)
    rfmappend=np.vstack((rfmappend.reshape(-1,1),rfmtemp.reshape(-1,1)))

rfmappend=rfmappend.flatten()
rfmappend['rec'][np.where(rfmappend['freq']==0)]=-1

#test hids: 
print(rfmappend[np.where(rfmappend['hid']==26831937)])

#save rfm
import os
os.chdir('/home/dmf/python/npz/')
np.savez_compressed('rfmhid.npz',rfm=rfmappend)

