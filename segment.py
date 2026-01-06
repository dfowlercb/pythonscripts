import numpy as np
from numpy.lib import recfunctions as rfn
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import hstack
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
np.set_printoptions(suppress=True)


#LOAD FILES AND CREATE FIELDS
l23=np.load('/home/dfowler/npz/l23.npz')['arr_0']
l23=np.sort(l23,order='CM_ID')
church=np.loadtxt('/home/dfowler/csv/ministryflags2.csv',dtype=[('cid','i8')],delimiter=',',encoding='latin1',usecols=0,skiprows=1)
church=np.sort(church,order='cid')

parts=np.loadtxt('/home/dfowler/csv/parts.txt',dtype=[('part' ,'U13' ), ('title','U100'), ('cat' ,'U9' ), ('bg' , 'U23')],delimiter=',',encoding='latin1',usecols=(6,8,11,12),skiprows=1)
parts=np.sort(parts,order='part')
idx=np.searchsorted(parts['part'],l23['SKU'])
CAT=parts['cat'][idx]
bg=parts['bg'][idx]
DATE=l23['ORDER_DATE'].astype('datetime64[D]')
SKUTITLE=parts['title'][idx]
tiers=np.char.split(CAT,sep='-')
t1 = np.array([value[0] for value in tiers])
t2 = np.array([value[1] for value in tiers])
t3 = np.array([value[2] for value in tiers])
l23=rfn.append_fields(l23,('DATE','SKU_TITLE','T1','T2','T3','BG'),(DATE,SKUTITLE,t1,t2,t3,bg),usemask=False)
l23=l23[np.where(l23['T1'] !='20')]
print(np.unique(l23['CM_ID']).shape)
#CONFIGURE SELECTS

#CHURCHES
#Keep
#idx=np.isin(l23['CM_ID'],church['cid'])
#Remove
idx=np.isin(l23['CM_ID'],church['cid'],invert=True)
#Select
l23=l23[idx]
print(np.unique(l23['CM_ID']).shape)

#HOMESCHOOL
subset=np.where(l23['BG']=='HOMESCHOOLING')[0]
cid=np.unique(l23['CM_ID'][subset])
#Keep
#idx=np.isin(l23['CM_ID'],cid)
#Remove
idx=np.isin(l23['CM_ID'],cid,invert=True)
#Select
l23=l23[idx]
print(np.unique(l23['CM_ID']).shape)

#GIFT
subset=np.where(l23['T1']=='05')[0]
cid=np.unique(l23['CM_ID'][subset])
#Keep
idx=np.isin(l23['CM_ID'],cid)
#Remove
#idx=np.isin(l23['CM_ID'],cid,invert=True)
#Select
l23=l23[idx]
print(np.unique(l23['CM_ID']).shape)

#SPARSE MATRIX
enc = OneHotEncoder()

#PRIMARY FEATURE
enc_data=enc.fit_transform(l23['BG'].reshape(-1,1))
#enc_data=enc.fit_transform(l23['T2'].reshape(-1,1))
#enc_data=enc.fit_transform(l23['T3'].reshape(-1,1))
print(enc.categories_)

#MATRIX CREATION
data=l23['UNITS']*l23['UNIT_PRICE']
row=l23['CM_ID']
col=enc_data.indices
X=sparse.coo_matrix((data,(row,col))).tocsr()

#scaling
custidx=np.unique(l23['CM_ID'])
X_scaled = preprocessing.MaxAbsScaler().fit_transform(X[custidx])

#kmeans
n=8
kmeans=MiniBatchKMeans(n_clusters=n)
kmeansclus=kmeans.fit(X_scaled)
kmeans_predicted = kmeans.predict(X_scaled)
clusters=np.unique(kmeans_predicted,return_counts=True)
print(clusters)

#output
cid0=custidx[np.where(kmeans_predicted==0)]
output=rfn.unstructured_to_structured( np.sum(X[cid0],0))

for i in range(1,n):
    cid1=custidx[np.where(kmeans_predicted==i)]
    output=np.hstack((output,rfn.unstructured_to_structured( np.sum(X[cid1],0))))

output.dtype.names= np.hstack(enc.categories_).tolist()

#export
export=np.transpose(rfn.structured_to_unstructured(output))
export=np.vstack((clusters[1],export))
np.savetxt('/home/dfowler/documents/segment.txt',export,delimiter=',',header=','.join(output.dtype.names),fmt='%i')

for i in range(0,n):
    cid=custidx[np.where(kmeans_predicted==i)]
    np.savetxt('/home/dfowler/documents/segmentcid'+str(i)+'.txt',cid,fmt='%i')

#SUBCLUSTERS
for i in range(0,n):
    clusternum=i
    subclust=custidx[np.where(kmeans_predicted==clusternum)]
    subcluster=l23[np.isin(l23['CM_ID'],subclust)]
    #enc_data=enc.fit_transform(subcluster['T2'].reshape(-1,1))
    enc_data=enc.fit_transform(subcluster['SKU_TITLE'].reshape(-1,1))
    data=subcluster['UNITS']*subcluster['UNIT_PRICE']
    row=subcluster['CM_ID']
    col=enc_data.indices
    X=sparse.coo_matrix((data,(row,col))).tocsr()
    basket=np.sum(X,0).reshape(-1,1)
    index=np.argsort(-basket,0)
    basket=basket[index]
    basket=basket.reshape(-1,1)
    basket=basket[:25]
    header=np.concatenate(enc.categories_)
    header=header[np.squeeze(np.asarray(index))][:25]
    header=','.join(header)
    fname='/home/dfowler/documents/subcluster'+str(i)+'.txt'
    np.savetxt(fname,basket,delimiter=',',header=header,fmt='%3.2f')



