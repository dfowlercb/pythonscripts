import numpy as np
import os
from numpy.lib import recfunctions as rfn

os.chdir('/home/dfowler/npz')
parts=np.loadtxt('/home/dfowler/csv/parts.txt',dtype=[('part' ,'U13' ), ('pub','U4'), ('cat' ,'U9' )],delimiter=',',encoding='latin1',usecols=(6,10,11),skiprows=1)

l23=np.load('l23.npz')['arr_0']
l24=np.load('l24.npz')['arr_0']
l24a=np.load('l25.npz')['arr_0']
arr_0=rfn.stack_arrays((l23,l24,l24a),usemask=False)
arr_0.shape

#add parts fields
parts=np.sort(parts,order='part')
idx=np.searchsorted(parts['part'],arr_0['SKU'])
CAT=parts['cat'][idx]
PUB=parts['pub'][idx]
tiers=np.char.split(CAT,sep='-')
t1 = np.array([value[0] for value in tiers])
t2 = np.array([value[1] for value in tiers])
t3 = np.array([value[2] for value in tiers])
arr_0=rfn.append_fields(arr_0,('T1','T2','T3','PUB'),(t1,t2,t3,PUB),usemask=False)

select=arr_0.copy()
select.shape

#ministry flag select
flag=np.loadtxt('/home/dfowler/csv/ministryflags2.csv',dtype=[('CID' ,'i8' ), ('FLAG' ,'i8' ),],delimiter=',',encoding='latin1',usecols=(0,2),skiprows=1)
flag=np.unique(flag)
flag=np.sort(flag,order='CID')

idx=np.isin(select['CM_ID'],flag['CID'])
select=select[idx]
select.shape

#female gender select
idx=select['GENDER']=='F'
select=select[idx]
select.shape

#APOLOGETICS SELECT
idx=select['T2']=='APO'
select=select[idx]

#theology select
idx=np.logical_or.reduce((select['pub']=='THE', select['pub']=='CMM', select['pub']=='CHH'))
select[idx].shape
arr_0.shape

#female gender select
idx=arr_0['GENDER']=='F'
arr_0=arr_0[idx]
arr_0.shape

#combine selects into one






