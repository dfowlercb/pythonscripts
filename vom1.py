import numpy as np
import os
from scipy import sparse
from scipy.stats import rankdata
from sklearn.preprocessing import scale

os.chdir('/home/dmf/python/npz')
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

def load(monthlevel,*features):
    import os
    os.chdir('/home/dmf/python/npz')
    fname=''.join([monthlevel,'hilo.npz'])
    arrays=np.load(fname,allow_pickle=True)
    for x in np.arange(len(features)):
        globals()[features[x]]=arrays[features[x]]

load('olfeb', 'cid','oidol','date','state')
load('llfeb','oid','sku','units','price','parts')

#EXCLUSIONS

#NO ORDERS PRIOR TO 2017
datemin = np.datetime64('2017-01-01','D').astype(int)
def scope(idx,*args):
    """
    Reduces the scope of an array(s) based on an index.
    
    Parameters:
        idx: A statement resulting in an index.
        args: The names of arrays to which an index is applied.
    
    Returns: All of the supplied arrays are reduced in scope by the index.
    """
    
    for i in np.arange(len(args)):
        globals()[args[i]]=globals()[args[i]][idx]

scope(np.where ( date >= datemin),'cid','oidol','date','state')

#NO RECENCY > 3 YEARS
recency =  np.array( [ cid, date ] )
idx = np.lexsort (( -date  , cid )) 
recency = recency [ : , idx ]
maxdate = np.searchsorted( recency[0], np.unique(cid) )
recency = recency[ : , maxdate ]
cutoff = np.datetime64('2019-01-31','D').astype(int)
recency = recency[:,np.where(recency[1] >= cutoff)[0]]

scope(np.isin (cid , recency[0]),'cid','oidol','state')

#NO INTERNATIONAL
scope(np.where( state != 57 ),'cid','oidol')
scope(np.isin ( oid, oidol ),'oid','sku','units','price')

#NO CATEGORY 20
cat20  = np.where ( parts[:,3] == '20' )
scope(np.isin (sku , cat20 , invert = True ),'oid','sku','units','price')
scope(np.isin (oidol, np.unique(oid) ),'cid','oidol')

#CONTROL GROUP VS TEST GROUP
responder=np.genfromtxt('/home/dmf/R/projects/vom/vomemailords.05122021.txt',delimiter='\t',usecols=(1,2),names=True,dtype='int')
control = np.isin (oidol , responder['ORDER'] , invert = True )    
test = np.isin (oidol , responder['ORDER'] )

cid_ctl = cid [ control ]
oidol_ctl = oidol [ control ]

idx = np.isin ( oid, oidol_ctl )
oid_ctl = oid [ idx ]
sku_ctl = sku [ idx ]
units_ctl = units [ idx ] 
price_ctl = price [ idx ]
ll_link_ctl = np.searchsorted ( oidol_ctl , oid_ctl) 

cid_tst = cid [ test ]
oidol_tst = oidol [ test ]

idx = np.isin ( oid, oidol_tst )
oid_tst = oid [ idx ]
sku_tst = sku [ idx ]
units_tst = units [ idx ] 
price_tst = price [ idx ]
ll_link_tst = np.searchsorted ( oidol_tst , oid_tst)

#FIRST PASS COMPARISONS
rows = np.repeat(0,sku_ctl.size)
cols = parts[sku_ctl,8].astype(int)
data = units_ctl * price_ctl
mtx_ctl=sparse.coo_matrix((data,(rows,cols))).toarray()
mtx_ctl = mtx_ctl / np.sum (mtx_ctl) *100

rows = np.repeat(0,sku_tst.size)
cols = parts[sku_tst,8].astype(int)
data = units_tst * price_tst
mtx_tst=sparse.coo_matrix((data,(rows,cols))).toarray()
mtx_tst = mtx_tst / np.sum (mtx_tst) *100

np.array( [ mtx_ctl , mtx_tst ] )

#SKU MULTIPLES
rows = sku_ctl
cols = np.repeat(0,sku_ctl.size)
data = units_ctl * price_ctl
shape = tuple (( parts[:,0].size , 1 ))
mtx_ctl = sparse.coo_matrix((data,(rows,cols)),shape=shape).toarray()
mtx_ctl = mtx_ctl / np.sum (mtx_ctl) *100
rank_ctl = rankdata(-mtx_ctl,method='dense')

rows = sku_tst
cols = np.repeat(0,sku_tst.size)
data = units_tst * price_tst
shape = tuple (( parts[:,0].size , 1 ))
mtx_tst = sparse.coo_matrix((data,(rows,cols)),shape=shape).toarray()
mtx_tst = mtx_tst / np.sum (mtx_tst) *100
rank_tst = rankdata(-mtx_tst,method='dense')

multiple= np.divide(mtx_tst, mtx_ctl) 

#BUYER GROUP EXCLUSIONS
cidctlnum = np.unique( cid_ctl, return_inverse=True)
rows = cidctlnum[1][ll_link_ctl]
cols = parts [ sku_ctl , 7 ].astype(int)
data = units_ctl * price_ctl
shape = tuple (( cidctlnum[0] .size , np.unique ( parts [ sku_ctl , 7]).size ))
mtx_bg = sparse.coo_matrix((data,(rows,cols)),shape=shape).toarray()
mtx_bg_key = np.unique(np.array((parts[:,2],parts[:,7])),axis=1).T
exclude = np.where(np.isin(mtx_bg_key[:,0], np.array(['CHURCH INTEREST','BIBLES','ACADEMIC','GIFTS','CLOTHING','KIDS','HOMESCHOOLING','MUSIC']))==True)
idx = np.argmax(mtx_bg,axis=1)
cid_keep = cidctlnum[0][np.isin (idx , exclude, invert = True)]

idx = np.isin(cid_ctl , cid_keep)
cid_ctl = cid_ctl [ idx ]
oidol_ctl = oidol_ctl [ idx ]

idx = np.isin ( oid, oidol_ctl )
oid_ctl = oid [ idx ]
sku_ctl = sku [ idx ]
units_ctl = units [ idx ] 
price_ctl = price [ idx ]
ll_link_ctl = np.searchsorted ( oidol_ctl , oid_ctl) 

#CATEGORY SELECTS
def t2t3(*code):
    idx=np.empty(shape=0,dtype=int)
    for code in code:
        idx=np.append(idx,np.where(np.char.find(parts[:,1],code)>-1))
    return(np.unique(idx))

def endcode(*code):
    idx=np.empty(shape=0,dtype=int)
    for code in code:
        idx=np.append(idx,np.where(np.char.endswith(parts[:,1],code)==True))
    return(np.unique(idx))

keepcat0=t2t3('APO-GN','AUD-AL','AUD-BO','AUD-DS','AUD-FC','AUD-FN','AUD-GN','AUD-IN','AUD-SI','AUD-WN','BBS-NT','BBS-OT','BIB-DQ','BIO-GN','BIO-HI','BIO-IN','CCL-FS','CCL-GN','CCL-PY','CMM-NT','CSB-SB','DRA-BL','DRA-FS','ESV-RF','ESV-SC','KJV-RF','ISS-GN','ISS-SX','ISS-US','JES-BL','MNC-CD','NKJ-RF','NKJ-LG','NLT-SB','PRO-GN','THE-DZ','THE-GN','THE-NT','THE-RF','THE-SY','THE-TG')
keepcat1=endcode ('LW','CZ','RR','MS','ME','RJ')
keepcat=np.union1d(keepcat0,keepcat1)

idx = np.isin (sku_ctl, keepcat)

oid_ctl = oid_ctl [ idx ]
sku_ctl = sku_ctl [ idx ]
units_ctl = units_ctl [ idx ] 
price_ctl = price_ctl [ idx ]

idx = np.isin ( oidol_ctl, oid_ctl )
cid_ctl = cid_ctl [ idx ]
oidol_ctl = oidol_ctl [ idx ]

#PRIOR YEAR SELECTS
a=np.genfromtxt('/home/dmf/R/projects/vom/sent2019.csv',delimiter=',',dtype=int)
b=np.genfromtxt('/home/dmf/R/projects/vom/sent2020.csv',delimiter=',',dtype=int)
c=np.genfromtxt('/home/dmf/R/projects/vom/sent2021.csv',delimiter=',',dtype=int)
remove=np.hstack((a,b,c))

idx = np.isin(cid_ctl , remove , invert = True )
cid_ctl = cid_ctl [ idx ]
oidol_ctl = oidol_ctl [ idx ]

idx = np.isin ( oid_ctl, oidol_ctl )
oid_ctl = oid_ctl [ idx ]
sku_ctl = sku_ctl [ idx ]
units_ctl = units_ctl [ idx ] 
price_ctl = price_ctl [ idx ]

ll_link_ctl = np.searchsorted ( oidol_ctl , oid_ctl) 

#SCORING
cid_ctl_key = np.unique ( cid_ctl , return_inverse = True )

#multiple * sales by cid
rows = cid_ctl_key[1][ll_link_ctl]
cols = np.repeat (0 , rows.size)
data = (units_ctl * price_ctl) * multiple[sku_ctl].flatten()  
shape = tuple (( np.unique(cid_ctl).size , 1 ))
scr1 = sparse.coo_matrix((data,(rows,cols)),shape=shape).toarray()

 #multiple * units by cid
rows = cid_ctl_key[1][ll_link_ctl]
cols = np.repeat (0 , rows.size )
data = units_ctl * multiple[sku_ctl].flatten()
shape = tuple (( np.unique(cid_ctl).size  , 1 ))
scr2 = sparse.coo_matrix((data,(rows,cols)),shape=shape).toarray()

 #weighted average (multiple by units) by cid
#find the sum of all the variables multiplied by their weight, then divide by the sum of the weights.

rows = cid_ctl_key[1][ll_link_ctl]
cols = np.repeat (0, rows.size)
data = units_ctl
shape = tuple (( np.unique(cid_ctl).size  , 1 ))
scr3a = sparse.coo_matrix((data,(rows,cols)),shape=shape).toarray()
scr3 = scr2 / scr3a

scr1 = scale ( scr1 )
scr2 = scale ( scr2 )
scr3 = scale ( scr3 )

scores = np.hstack (( scr1 , scr2  , scr3 ))
scores = np.hstack (( scores, np.mean ( scores, axis=1).reshape(-1,1) ))

export = np.array ((cid_ctl_key[0] , scores[:,3] ))
export = export[:,(-export[1]).argsort()]
export = np.vstack (( export , np.repeat(1 , export[1].size) ))

export [2 : , np.where( (export[1]>np.percentile(export[1],20)) & (export[1] <= np.percentile(export[1],40)) )[0] ] =  2
export [2 : , np.where( (export[1]>np.percentile(export[1],40)) & (export[1] <= np.percentile(export[1],60)) )[0] ] =  3
export [2 : , np.where( (export[1]>np.percentile(export[1],60)) & (export[1] <= np.percentile(export[1],80)) )[0] ] =  4
export [2 : , np.where( (export[1]>np.percentile(export[1],80)) & (export[1] <= np.percentile(export[1],100)) )[0] ] =  5

compareR = np.genfromtxt ( '/home/dmf/Documents/vomscores.txt' , delimiter='\t', dtype=int, names = True)

np.intersect1d(export[0].astype(int) , compareR['CID']).size / export[0].size

np.intersect1d(export[0][np.where(export[2]==5)].astype(int) , compareR['CID'][np.where(compareR['GROUP']==5)]).size / export[0][np.where(export[2]==5)].size
np.intersect1d(export[0][np.where(export[2]==4)].astype(int) , compareR['CID'][np.where(compareR['GROUP']==4)]).size / export[0][np.where(export[2]==4)].size
np.intersect1d(export[0][np.where(export[2]==3)].astype(int) , compareR['CID'][np.where(compareR['GROUP']==3)]).size / export[0][np.where(export[2]==3)].size
np.intersect1d(export[0][np.where(export[2]==2)].astype(int) , compareR['CID'][np.where(compareR['GROUP']==2)]).size / export[0][np.where(export[2]==2)].size
np.intersect1d(export[0][np.where(export[2]==1)].astype(int) , compareR['CID'][np.where(compareR['GROUP']==1)]).size / export[0][np.where(export[2]==1)].size

