import numpy as np
x=np.load('/home/dfowler/npz/l24.npz')['arr_0']
sku=np.unique(x['SKU'],return_inverse=True)
date=np.unique(x['ORDER_DATE'],return_inverse=True)
lowcost=np.unique(arr22['lowcost'],return_inverse=True)


row=oid[1]
col=classify[1]
shape=tuple((oid[0].size,classify[0].size))
data=arr22['wt']
csr=sparse.coo_matrix((data,(row,col)),shape=shape).tocsr()
print(csr.shape)
weight=csr.toarray()
