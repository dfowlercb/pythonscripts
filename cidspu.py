import numpy as np
ol=np.load('/home/dmf/python/npz/olhigh.npz')
cid=ol['cid']
cid[np.where(cid==-1)]=0
date[np.where(date<0)]=0
date=ol['date']
ll=np.load('/home/dmf/python/npz/llhigh.npz')
units=ll['units']
units=units.astype(int)
price=ll['price']
ll_link=ll['ll_link']
from scipy.sparse import coo_matrix
price2=np.unique(price,return_inverse=True)
sales=units*price
col=np.repeat(0,len(sales))
x=coo_matrix((sales,(cid[ll_link],col)))
y=coo_matrix((units,(cid[ll_link],col)))
z=x/y
x=x.tocsr()
spu=z[cid]
spu=np.squeeze(np.asarray(spu))
a=cid[np.unique(cid,return_index=True)[1]]
b=spu[np.unique(cid,return_index=True)[1]]

date=date.astype(int)
a=coo_matrix((units,(cid[ll_link],date[ll_link])))
i=np.argmax(a,axis=1)
i[33500001]

