import numpy as np
from numpy.lib import recfunctions as rfn
from scipy import sparse
from pyfunc.arrfunc import stackarr as stk

#arr_0=np.load('/home/dfowler/npz/hi1997_2001.npz')['arr_0']
#arr_0=stk(arr_0,'hi2002_2006')
#arr_0=stk(arr_0,'hi2007_2011')
#arr_0=stk(arr_0,'hi2012_2016')
#arr_0=stk(arr_0,'h17')
#arr_0=stk(arr_0,'h18')
arr_0=np.load('/home/dfowler/npz/h2018_2023.npz')['arr_0']

#load updated parts file
parts=np.loadtxt('/home/dfowler/csv/parts.txt',usecols=(6,11,12),delimiter=',',skiprows=1,dtype='str',encoding='latin1')
parts=rfn.unstructured_to_structured(parts)
parts.dtype.names=('sku','cat','bg')
tiers=np.char.partition(parts['cat'],'-')
t1=tiers[:,0].reshape(-1,1)
tiers=np.char.partition(tiers[:,2],'-')
t2=tiers[:,0].reshape(-1,1)
t3=tiers[:,2].reshape(-1,1)
del tiers
parts=rfn.append_fields(parts,('t1','t2','t3'),(t1,t2,t3),usemask=False)
t1num=np.unique(parts['t1'],return_inverse=True)
t2num=np.unique(parts['t2'],return_inverse=True)
t3num=np.unique(parts['t3'],return_inverse=True)
bgnum=np.unique(parts['bg'],return_inverse=True)
parts=rfn.append_fields(parts,('t1num','t2num','t3num','bgnum'),(t1num[1],t2num[1],t3num[1],bgnum[1]),usemask=False)
parts=np.sort(parts,order='sku')

#testing
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})
years = arr_0['ORDER_DATE'].astype('datetime64[D]').astype('datetime64[Y]').astype(int)
years[years<0]=0
data=arr_0['UNITS']*arr_0['UNIT_PRICE']
row=np.repeat(0,data.size)
coo=sparse.coo_matrix((data,(row,years))).toarray()
print(coo.T)

#preparatory
cidnum=np.unique(arr_0['CM_ID'],return_inverse=True)
bu=np.repeat(0,cidnum[0].size)
skunum=np.searchsorted(parts['sku'],arr_0['SKU'])
hms=np.where(t2num[0]=='HMS')[0].item()

#t2 sparse matrix
row=cidnum[1]
col=parts['t2num'][skunum]
shape=tuple((cidnum[0].size,t2num[0].size))
coo=sparse.coo_matrix((data,(row,col)),shape=shape).tocsr()
print(coo.shape)

#homeschool
  #hms>.5
x=np.sum(coo[:,hms],axis=1)/np.sum(coo,axis=1)
bu[np.where(x>.5)[0]]=1

 #hms = max
x=np.argmax(coo,axis=1)
bu[np.where(x==hms)[0]]=1

 #church
church=np.unique(arr_0['CM_ID'][np.where(arr_0['CHUR_TAX_CID']>0)[0]])
church=np.searchsorted(cidnum[0],church)
buopen=np.where(bu==0)[0]
idx=np.intersect1d(church,buopen)
bu[idx]=2

 #professional
idx=np.isin(t2num[0],np.array(['APO','BBS','BRF','BST','CHH','CHR','CMM','CSL','CUR','DSS','EVA','GNR','INS','LAN','LEA','PHI','PRE','PSY','THE','VBS']))
idx=np.where(idx==True)[0]
pro1=np.sum(coo[:,idx],axis=1)

#t3 sparse matrix
row=cidnum[1]
col=parts['t3num'][skunum]
shape=tuple((cidnum[0].size,t3num[0].size))
coo=sparse.coo_matrix((data,(row,col)),shape=shape).tocsr()
print(coo.shape)
idx=np.isin(t3num[0],np.array(['GA','PP','EM','BP','LD','YM','PR']))
idx=np.where(idx==True)[0]
pro2=np.sum(coo[:,idx],axis=1)

pro3=pro1+pro2
pro=cidnum[0][np.where(pro3>=100)[0]]
pro=np.searchsorted(cidnum[0],pro)
buopen=np.where(bu==0)[0]
idx=np.intersect1d(pro,buopen)
bu[idx]=3

#business unit sales by year
data=arr_0['UNITS']*arr_0['UNIT_PRICE']
row=bu[cidnum[1]]
col=years
shape=(np.unique(bu).size,np.max(years)+1)
coo=sparse.coo_matrix((data,(row,col)),shape=shape).tocsr()
coo=coo[:,27:].toarray()
np.savetxt('/home/dfowler/documents/x.txt',coo,delimiter='\t',fmt='%f')

#business unit cid export
cidbu=np.array((cidnum[0],bu))
cidbu=np.unique(cidbu,axis=1)
cidbu=cidbu[:,cidbu[0]>0]
np.savetxt('/home/dfowler/documents/cidbudec22.txt',cidbu.T,fmt='%i',delimiter='\t')

#consumer subcategories
bu=bu[cidnum[1]]
bg=parts['bgnum'][skunum]
t1=parts['t1num'][skunum]
label=np.zeros(len(arr_0),dtype=int)
arr_0=rfn.drop_fields(arr_0,('ORDER_NO','HH_ID','CHUR_TAX_CID','CHUR_TAX_HID','SKU'))
arr_0=rfn.append_fields(arr_0,('bu','bg','t1','label'),(bu,bg,t1,label),usemask=False)
arr_0=arr_0[arr_0['bu']==0]
arr_0=arr_0[arr_0['CM_ID']>=0]

sales=arr_0['UNITS']*arr_0['UNIT_PRICE']
row=arr_0['CM_ID']
col=arr_0['bg']
bgmtx = sparse.coo_matrix((sales,(row,col)))
bgmtx = bgmtx.tocsr()
bgmtx=bgmtx.toarray()/np.nansum(bgmtx.toarray(),axis=1, keepdims=True)

col=arr_0['t1']
t1mtx = sparse.coo_matrix((sales,(row,col)))
t1mtx = t1mtx.tocsr()
t1mtx=t1mtx.toarray()/np.nansum(t1mtx.toarray(),axis=1, keepdims=True)

column=np.where(bgnum[0]=='BIBLES')[0].item()
i=np.where(bgmtx[:,column]>.5)[0]
arr_0['label'][np.isin(arr_0['CM_ID'],i)]=1

column=np.where(t1num[0]=='09')[0].item()
i=np.where(t1mtx[:,column]>.5)[0]
arr_0['label'][np.isin(arr_0['CM_ID'],i)]=2

column=np.where(bgnum[0]=='GIFTS')[0].item()
i=np.where(bgmtx[:,column]>.5)[0]
arr_0['label'][np.isin(arr_0['CM_ID'],i)]=3

column=np.where(bgnum[0]=='KIDS')[0].item()
i=np.where(bgmtx[:,column]>.5)[0]
arr_0['label'][np.isin(arr_0['CM_ID'],i)]=4

consumerbu=np.array([ arr_0['CM_ID'], arr_0['label'] ])
consumerbu=np.unique(consumerbu,axis=1)
np.savetxt('/home/dfowler/documents/consumerbudec22.txt',consumerbu.T,fmt='%i',delimiter='\t')

