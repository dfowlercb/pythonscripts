import numpy as np
from scipy import sparse
import os
from numpy import genfromtxt
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

#LOAD FILES
os.chdir('/home/dmf/python/npz')
 #order level   
arrays=np.load('olfebhilo.npz',allow_pickle=1)

oidol=arrays['oidol']
cid=arrays['cid']
date=arrays['date']
hid=arrays['hid']
churchcid=arrays['churchcid']
churchhid=arrays['churchhid']
channel=arrays['channel']
channelkey=arrays['channelkey']
pay=arrays['pay']
paykey=arrays['paykey']
zipcode=arrays['zipcode']
zipkey=arrays['zipkey']
state=arrays['state']
statekey=arrays['statekey']
new=arrays['new']
entry=arrays['entry']
entrykey=arrays['entrykey']
gender=arrays['gender']
genderkey=arrays['genderkey']
member=arrays['member']
memberkey=arrays['memberkey']

 #line level
arrays=np.load('llfebhilo.npz',allow_pickle=1)
oid=arrays['oid']
sku=arrays['sku']
units=arrays['units']
price=arrays['price']
pah=arrays['pah']
tax=arrays['tax']
cost=arrays['cost']
status=arrays['status']
statuskey=arrays['statuskey']
shipped=arrays['shipped']
parts=arrays['parts']
oldskukey=arrays['oldskukey']
ll_link=arrays['ll_link']

#date magic
date = date.astype('datetime64[D]')
years = date.astype('datetime64[Y]') 
months= (date.astype('datetime64[M]')-date.astype('datetime64[Y]')+1).astype('int64')
weeks = (date.astype('datetime64[W]')-date.astype('datetime64[Y]')+1).astype('int64')
weekdays = (date.astype('datetime64[D]').astype('int64') -4) % 7 +1
monthdays =  date.astype('datetime64')-date.astype('datetime64[M]') + 1
yeardays = date.astype('datetime64')-date.astype('datetime64[Y]') + 1

years=np.unique(years,return_inverse=1)
months=np.unique(months,return_inverse=1)
weeks=np.unique(weeks,return_inverse=1)
weekdays=np.unique(weekdays,return_inverse=1)
monthdays=np.unique(monthdays,return_inverse=1)
yeardays=np.unique(yeardays,return_inverse=1)
 #shipdate
date2 = shipped.astype('datetime64[D]')
date2[np.where( (date2>=np.datetime64('2021-01-05') ) & (date2<=np.datetime64('2021-02-01') ) )]



years2 = date2.astype('datetime64[Y]').astype('int64')
months2= (date2.astype('datetime64[M]')-date2.astype('datetime64[Y]')+1).astype('int64')

years2=np.unique(years2,return_inverse=1)
months2=np.unique(months2,return_inverse=1)

#sales by year
data=units*price
row=np.repeat(0,data.size)
col=years[1][ll_link]
coo=sparse.coo_matrix((data,(row,col))).toarray()
print(coo.T)

#assign business units
data=units*price
cid1=cid[ll_link]
cidnum=np.unique(cid1,return_inverse=1)
years1=years[1][ll_link]
churchcid1=churchcid[ll_link]
bu=np.repeat(0,cid1.size)

 #homeschool
hms=parts[np.where(parts[:,4]=='HMS')[0],9].astype(int)[:1]

  #sparse matrix
row=cidnum[1]
col=parts[sku,9].astype(int)
shape=tuple((cidnum[0].size,np.unique(parts[:,9]).size))
coo=sparse.coo_matrix((data,(row,col)),shape=shape).tocsr()

  #hms>.5
x=np.sum(coo[:,hms],axis=1)/np.sum(coo,axis=1)
buhms1=cidnum[0][np.where(x>.5)[0]]
bu[np.isin(cid1,buhms1)]=1

 #hms = max
x=np.argmax(coo,axis=1)
buhms2=cidnum[0][np.where(x==hms)[0]]
bu[np.isin(cid1,buhms2)]=1

 #church
church=cid[np.where(churchcid==1)[0]]
church=church[np.isin(church,buhms1,invert=1)]
church=church[np.isin(church,buhms2,invert=1)]
bu[np.isin(cid1,church)]=2

 #professional
row=cidnum[1]
col=parts[sku,9].astype(int)
shape=tuple((cidnum[0].shape[0],np.max(parts[:,9].astype(int))+1))
coo=sparse.coo_matrix((data,(row,col)),shape=shape).tocsr()
i=np.unique((parts[np.isin(parts[:,4],np.array(['APO','BBS','BRF','BST','CHH','CHR','CMM','CSL','CUR','DSS','EVA','GNR','INS','LAN','LEA','PHI','PRE','PSY','THE','VBS'])),9]).astype(int))
pro1=np.sum(coo[:,i],axis=1)

row=cidnum[1]
col=parts[sku,10].astype(int)
shape=tuple((cidnum[0].shape[0],np.max(parts[:,10].astype(int))+1))
coo=sparse.coo_matrix((data,(row,col)),shape=shape).tocsr()
i=np.unique(parts[np.isin(parts[:,5],np.array(['GA','PP','EM','BP','LD','YM','PR'])),10].astype(int))
pro2=np.sum(coo[:,i],axis=1)

pro3=pro1+pro2
pro=cidnum[0][np.where(pro3>=100)[0]]
pro=pro[np.isin(pro,buhms1,invert=1)]
pro=pro[np.isin(pro,buhms2,invert=1)]
pro=pro[np.isin(pro,church,invert=1)]
bu[np.isin(cid1,pro)]=3

#business unit sales by year
data=units*price
row=bu
col=years1
shape=(np.unique(bu).size,np.unique(years1).size)
coo=sparse.coo_matrix((data,(row,col)),shape=shape).tocsr()
coo.toarray()

#business unit cid export
cidbu=np.array((cid1,bu))
cidbu=np.unique(cidbu,axis=1)
os.chdir('/home/dmf/Documents')
np.savetxt('cidbu.txt',cidbu.T,fmt='%i',delimiter='\t')

#business unit sales by shipdate
shipdate = shipped.astype('datetime64[D]')
yearsshipdate = shipdate.astype('datetime64[Y]') 
yearsshipdate = np.unique(yearsshipdate,return_inverse=1)

data=units*price
row=bu
col=yearsshipdate[1]
shape=(np.unique(bu).size,np.unique(yearsshipdate[1]).size)
coo=sparse.coo_matrix((data,(row,col)),shape=shape).tocsr()
coo.toarray()


#cohort
cidnum=np.unique(cid,return_inverse=1)
cohort=sparse.lil_matrix((cidnum[0].size,years[0].size),dtype='int64')
cohort[cidnum[1],years[1]]=years[0][years[1]]
cohort=cohort.todense()
cohort[np.where(cohort==0)]=100
cohort=np.amin(cohort,axis=1)
cohort=cohort.astype('datetime64[Y]')

row=cohort[cidnum[1]][ll_link]

#3D ARRAY
data=units*price
datefilter=np.where(date[ll_link].astype('datetime64[Y]')==np.datetime64(2021-1970,'Y'))
cid1=cid[ll_link][datefilter]
cidnum=np.unique(cid1,return_inverse=1)
years1=years[1][ll_link][datefilter]
data1=data[datefilter]
sku1=sku[datefilter]

I=cidnum[1]
J=parts[parts[sku1],7].astype(int)
V=data1
coo=sparse.coo_matrix((V,(I,J))).tocsr()
    
#orders by category with date filter
row=np.unique(oid,return_inverse=1)
col=parts[parts[sku],7].astype(int)
coo=sparse.coo_matrix((data[filterdate],(row[1][filterdate],col[filterdate])))

sparse.save_npz('/home/dmf/python/npz/coo_t1.npz', coo)

#JEWISH
os.chdir('/home/dmf/csv')
parts2=genfromtxt('parts.txt',skip_header=1,delimiter=',',dtype='str',encoding='latin1',usecols=(6,8),converters={6: lambda s: str(s or ''),8: lambda s: str(s or '')})
parts2=parts2[parts2[:,0].argsort(),]
np.array_equal(parts[:,0],parts2[:,0])

jewish=np.where( (parts[:,4]=='JUD') | ( np.isin(parts[:,5],np.array(['JE','RJ']))))
ot=np.where( parts[:,5]=='OT')
x1=np.nonzero(np.char.find(parts2[:,1], 'Haggadah')!=-1)
x2=np.nonzero(np.char.find(parts2[:,1], 'Hanukkah')!=-1)
x3=np.nonzero(np.char.find(parts2[:,1], 'Hasidic')!=-1)
x4=np.nonzero(np.char.find(parts2[:,1], 'Hebraic')!=-1)
x5=np.nonzero(np.char.find(parts2[:,1], 'Holocaust')!=-1)
x6=np.nonzero(np.char.find(parts2[:,1], 'Israel')!=-1)
x7=np.nonzero(np.char.find(parts2[:,1], 'Jewish')!=-1)
x8=np.nonzero(np.char.find(parts2[:,1], 'Judaism')!=-1)
x9=np.nonzero(np.char.find(parts2[:,1], 'Midrash')!=-1)
x10=np.nonzero(np.char.find(parts2[:,1], 'Mishnah')!=-1)
x11=np.nonzero(np.char.find(parts2[:,1], 'Passover')!=-1)
x12=np.nonzero(np.char.find(parts2[:,1], 'Rabbi')!=-1)
x13=np.nonzero(np.char.find(parts2[:,1], 'Rosh Hashanah')!=-1)
x14=np.nonzero(np.char.find(parts2[:,1], 'Seder')!=-1)
x15=np.nonzero(np.char.find(parts2[:,1], 'Shabbat')!=-1)
x16=np.nonzero(np.char.find(parts2[:,1], 'Shofar')!=-1)
x17=np.nonzero(np.char.find(parts2[:,1], 'Talmud')!=-1)
x18=np.nonzero(np.char.find(parts2[:,1], 'Tanakh')!=-1)
x19=np.nonzero(np.char.find(parts2[:,1], 'Torah')!=-1)
x20=np.nonzero(np.char.find(parts2[:,1], 'Yom Kippur')!=-1)
x21=np.nonzero(np.char.find(parts2[:,1], 'Judaica')!=-1)
from functools import reduce
keyword=reduce(np.union1d,(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21))


I=parts[sku]
J=years[1][ll_link]
V=units*price
m1=sparse.coo_matrix((V,(I,J)),shape=(parts.shape[0],years[0].shape[0])).tocsr()
np.sum(m1[jewish[0],:],axis=0).T
np.sum(m1[ot[0],:],axis=0).T
np.sum(m1[keyword,:],axis=0).T
os.chdir('/home/dmf/Documents')
np.savetxt('x.txt',m1[keyword,:].toarray(),delimiter='\t')
np.savetxt('y.txt',parts2[keyword,],fmt='%s',delimiter='\t')

#number of order lines study
x,lines=np.unique(oid,return_counts=1)
np.array_equal(oidol,x)
date=date.astype('datetime64[D]')
years=date.astype('datetime64[Y]') 
years=np.unique(years,return_inverse=1)
data=np.repeat(1,lines.size)
linesmatrix=sparse.coo_matrix((data,(lines,years[1]))).toarray()
yeartotals=np.sum(linesmatrix,axis=0)
linesmatrixperc=linesmatrix/yeartotals
os.chdir('/home/dmf/Documents')
np.savetxt('linesmatrixperc.txt',linesmatrixperc,fmt='%i',delimiter='\t')

#whales first order to lifetime $
 #customer total spend matrix
rowkeycid=np.unique(cid[ll_link],return_inverse=1)
shape=(rowkeycid[0].size,1)
I=rowkeycid[1]
J=np.repeat(0,I.size)
V=units*price
cidsum=sparse.coo_matrix((V,(I,J)),shape=shape).tocsr()
 #order total matrix
rowkeyoid=np.unique(oid,return_inverse=1)
shape=(rowkeyoid[0].size,1)
I=rowkeyoid[1]
J=np.repeat(0,I.size)
V=units*price
oidsum=sparse.coo_matrix((V,(I,J)),shape=shape).tocsr()
 #find first order
    # sort with major key groups, minor key data
order = np.lexsort((oidol, cid))
customers = cid[order] 
orders = oidol[order]
  # construct an index which marks borders between groups
index = np.empty(len(customers), 'bool')
index[0] = True
index[1:] = customers[1:] != customers[:-1]

firstorder=np.array((customers[index],orders[index]))

 #cid oid matchup
newoid=np.isin(rowkeyoid[0],firstorder[1,:])
newoidsum=oidsum[newoid].toarray()

i=np.argsort(firstorder[1,:])
firstorder=firstorder[:,i]
i=np.searchsorted(firstorder[1,:],rowkeyoid[0][newoid])
firstorder=np.vstack((firstorder,newoidsum[i].T))

i=np.argsort(firstorder[0,:])
firstorder=firstorder[:,i]
i=np.searchsorted(firstorder[0,:],rowkeycid[0])

#create bins and aggregate
firstorder=np.vstack((firstorder,cidsum.toarray().T))
binfirst=firstorder[2,:].copy()
binall=firstorder[3,:].copy()

binfirst[binfirst<20]=19.99
binfirst[(binfirst>=20) & (binfirst<50)]=49.99
binfirst[(binfirst>=50) & (binfirst<100)]=99.99
binfirst[binfirst>=100]=100
rowkeybinfirst=np.unique(binfirst,return_inverse=1)
shape=tuple((rowkeybinfirst[0].size,1))
I=rowkeybinfirst[1]
J=np.repeat(0,I.size)
V=np.repeat(1,I.size)
binfirstagg=sparse.coo_matrix((V,(I,J)),shape=shape)
binfirstagg.toarray()

binall[binall<20]=19.99
binall[(binall>=20) & (binall<50)]=49.99
binall[(binall>=50) & (binall<100)]=99.99
binall[(binall>=100) & (binall<250)]=249.99
binall[(binall>=250) & (binall<500)]=499.99
binall[(binall>=500) & (binall<1000)]=999.99
binall[binall>=1000]=1000
rowkeybinall=np.unique(binall,return_inverse=1)
shape=tuple((rowkeybinfirst[0].size,rowkeybinall[0].size))
I=rowkeybinfirst[1]
J=rowkeybinall[1]
V=np.repeat(1,I.size)
binallagg=sparse.coo_matrix((V,(I,J)),shape=shape)
binallagg.toarray()

#sum bins by actual amounts
V=firstorder[3,:].copy()
allagg=sparse.coo_matrix((V,(I,J)),shape=shape)
allagg.toarray()

#min max aggregation
import numpy as np
def group_min(groups, data):
    # sort with major key groups, minor key data
    order = np.lexsort((data, groups))
    groups = groups[order] # this is only needed if groups is unsorted
    data = data[order]
    # construct an index which marks borders between groups
    index = np.empty(len(groups), 'bool')
    index[0] = True
    index[1:] = groups[1:] != groups[:-1]
    return data[index]

#max is very similar
def group_max(groups, data):
    order = np.lexsort((data, groups))
    groups = groups[order] #this is only needed if groups is unsorted
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    return data[index]
