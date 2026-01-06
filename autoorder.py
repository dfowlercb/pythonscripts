import numpy as np
from numpy.lib import recfunctions as rfn
from scipy import sparse

np.set_printoptions(suppress=True)

y_mindate=np.datetime64('2023-01-01').astype(int)
y_maxdate=np.datetime64('2023-02-28').astype(int)

x=np.load('/home/dfowler/npz/l22.npz')['arr_0']
y=np.load('/home/dfowler/npz/l23.npz')['arr_0']
y=np.delete(y,np.where(y['ORDER_DATE']<y_mindate))
y=np.delete(y,np.where(y['ORDER_DATE']>y_maxdate))

convert={31: lambda s: float(s.strip() or 0), 32: lambda s: float(s.strip() or 0),}
partskey=np.loadtxt('/home/dfowler/csv/parts.txt',delimiter=',',usecols=(6,10,11,16,31,32),converters=convert,dtype=[('SKU','U21'),('PUB','U4'),('CAT','U9'),('DISC','U1'),('RETAIL','f8'),('PRICE','f8')],encoding='latin1',skiprows=1)
partskey=np.sort(partskey,order='SKU')

#ADD FIELDS TO x
idx=np.searchsorted(partskey['SKU'],x['SKU'])
CAT=partskey['CAT'][idx]
tiers=np.char.split(CAT,sep='-')
t1 = np.array([value[0] for value in tiers])
t2 = np.array([value[1] for value in tiers])
t3 = np.array([value[2] for value in tiers])
DISC=partskey['DISC'][idx]
PUB=partskey['PUB'][idx]
RETAIL=partskey['RETAIL'][idx]
PRICE=partskey['PRICE'][idx]

x=rfn.append_fields(x,('T1','T2','T3','DISC','PUB','SKUNUM','RETAIL','PRICE'),(t1,t2,t3,DISC,PUB,idx,RETAIL,PRICE),usemask=False)

#ADD FIELDS TO y
idx=np.searchsorted(partskey['SKU'],y['SKU'])
CAT=partskey['CAT'][idx]
tiers=np.char.split(CAT,sep='-')
t1 = np.array([value[0] for value in tiers])
t2 = np.array([value[1] for value in tiers])
t3 = np.array([value[2] for value in tiers])
DISC=partskey['DISC'][idx]
PUB=partskey['PUB'][idx]
RETAIL=partskey['RETAIL'][idx]
PRICE=partskey['PRICE'][idx]

y=rfn.append_fields(y,('T1','T2','T3','DISC','PUB','SKUNUM','RETAIL','PRICE'),(t1,t2,t3,DISC,PUB,idx,RETAIL,PRICE),usemask=False)

#REMOVE DIGITAL, DISC and IFC
idx=np.isin(x['T3'],['DL','EB','DA','VD','DF','DW'])
x=np.delete(x,np.where(idx==True))
idx=np.where(x['T1']=='04')
x=np.delete(x,idx)
x=np.delete(x,np.where(x['DISC']!=''))
x=np.delete(x,np.where(x['PUB']=='IFC'))

#ANNUAL UNITS PER SKU
annual_units=np.bincount(x['SKUNUM'],x['UNITS'])
annual_units[annual_units>0].shape
annual_units[np.logical_and(annual_units>0,annual_units<=36)].shape

#prepare X, y in Time Series Format 
day=[]
def weekday(start,end,day):
    for num in np.arange(start,end):
        day.append(np.is_busday(num,day))
        return(day)


X=sparse.coo_matrix((x['UNITS'],(x['SKUNUM'],x['ORDER_DATE']))).tocsr()
target=np.bincount(y['SKUNUM'],y['UNITS'])

X_sum=np.sum(X,1)
idx=np.where(X_sum>0)[0]
X=X[idx]
target=target[idx]

#CREATE MODEL
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test= train_test_split(X,target)
clf=DecisionTreeClassifier()
clf.fit(X_train, y_train)
predictions=clf.predict(X_test)
print(accuracy_score(y_test, predictions))
np.savetxt('/home/dfowler/documents/y_test.csv',y_test)
np.savetxt('/home/dfowler/documents/predictions.csv',predictions)
