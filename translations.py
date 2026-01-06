import numpy as np
from numpy import genfromtxt
from datetime import datetime 
from numpy import char
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

with np.load('/home/dmf/python/npz/arr2012.npz',allow_pickle=1) as data:
    dates12=data['dates12']
    skus12=data['skus12']
    units12=data['units12']
    price12=data['price12']

with np.load('/home/dmf/python/npz/arr2013.npz',allow_pickle=1) as data:
    dates13=data['dates13']
    skus13=data['skus13']
    units13=data['units13']
    price13=data['price13']

with np.load('/home/dmf/python/npz/arr2014.npz',allow_pickle=1) as data:
    dates14=data['dates14']
    skus14=data['skus14']
    units14=data['units14']
    price14=data['price14']

with np.load('/home/dmf/python/npz/arr2015.npz',allow_pickle=1) as data:
    dates15=data['dates15']
    skus15=data['skus15']
    units15=data['units15']
    price15=data['price15']

with np.load('/home/dmf/python/npz/arr2016.npz',allow_pickle=1) as data:
    dates16=data['dates16']
    skus16=data['skus16']
    units16=data['units16']
    price16=data['price16']

with np.load('/home/dmf/python/npz/arr2017.npz',allow_pickle=1) as data:
    dates17=data['dates17']
    skus17=data['skus17']
    units17=data['units17']
    price17=data['price17']

with np.load('/home/dmf/python/npz/arr2018.npz',allow_pickle=1) as data:
    dates18=data['dates18']
    skus18=data['skus18']
    units18=data['units18']
    price18=data['price18']

with np.load('/home/dmf/python/npz/arr2019.npz',allow_pickle=1) as data:
    dates19=data['dates19']
    skus19=data['skus19']
    units19=data['units19']
    price19=data['price19']

with np.load('/home/dmf/python/npz/arr2020.npz',allow_pickle=1) as data:
    dates20=data['dates20']
    skus20=data['skus20']
    units20=data['units20']
    price20=data['price20']

with np.load('/home/dmf/python/npz/arr2021.npz',allow_pickle=1) as data:
    dates21=data['dates21']
    skus21=data['skus21']
    units21=data['units21']
    price21=data['price21']

with np.load('/home/dmf/python/npz/parts.npz',allow_pickle=1) as data:
    t1=data['t1']
    t2=data['t2']
    t1dict=data['t1dict'].T
    t2dict=data['t2dict'].T

dates=np.concatenate((dates12,dates13,dates14,dates15,dates16,dates17,dates18,dates19,dates20,dates21))
skus=np.concatenate((skus12,skus13,skus14,skus15,skus16,skus17,skus18,skus19,skus20,skus21))
units=np.concatenate((units12,units13,units14,units15,units16,units17,units18,units19,units20,units21))
price=np.concatenate((price12,price13,price14,price15,price16,price17,price18,price19,price20,price21))
sales=units*price

del(dates12,dates13,dates14,dates15,dates16,dates17,dates18,dates19,dates20,dates21)
del(skus12,skus13,skus14,skus15,skus16,skus17,skus18,skus19,skus20,skus21)
del(units12,units13,units14,units15,units16,units17,units18,units19,units20,units21)
del(price12,price13,price14,price15,price16,price17,price18,price19,price20,price21)


bibles=np.where(t1==3)
dates=dates[bibles]
skus=skus[bibles]
units=units[bibles]
price=price[bibles]
sales=sales[bibles]
t2=t2[bibles]

binsort=np.lexsort((dates,t2))
data=sales[binsort]
dates=dates[binsort]
t2=t2[binsort]
years = dates.astype('datetime64[Y]').astype(int) + 1970
months = dates.astype('datetime64[M]').astype(int) % 12 + 1
#days=(dates - dates.astype('datetime64[M]')).astype(int) + 1
bins=np.array((t2,years,months),dtype='U4')
bins[2][np.where(bins[2]=='1')]='01'
bins[2][np.where(bins[2]=='2')]='02'
bins[2][np.where(bins[2]=='3')]='03'
bins[2][np.where(bins[2]=='4')]='04'
bins[2][np.where(bins[2]=='5')]='05'
bins[2][np.where(bins[2]=='6')]='06'
bins[2][np.where(bins[2]=='7')]='07'
bins[2][np.where(bins[2]=='8')]='08'
bins[2][np.where(bins[2]=='9')]='09'

bins2=np.zeros(len(bins[0]),dtype='int64')
bins2[np.where( (bins[1]=='2012') & (np.isin(bins[2],['10','11','12'])) | (bins[1]=='2013') & (np.isin(bins[2],['01','02','03','04','05','06','07','08','09'])) )]=1
bins2[np.where( (bins[1]=='2013') & (np.isin(bins[2],['10','11','12'])) | (bins[1]=='2014') & (np.isin(bins[2],['01','02','03','04','05','06','07','08','09'])) )]=2
bins2[np.where( (bins[1]=='2014') & (np.isin(bins[2],['10','11','12'])) | (bins[1]=='2015') & (np.isin(bins[2],['01','02','03','04','05','06','07','08','09'])) )]=3
bins2[np.where( (bins[1]=='2015') & (np.isin(bins[2],['10','11','12'])) | (bins[1]=='2016') & (np.isin(bins[2],['01','02','03','04','05','06','07','08','09'])) )]=4
bins2[np.where( (bins[1]=='2016') & (np.isin(bins[2],['10','11','12'])) | (bins[1]=='2017') & (np.isin(bins[2],['01','02','03','04','05','06','07','08','09'])) )]=5
bins2[np.where( (bins[1]=='2017') & (np.isin(bins[2],['10','11','12'])) | (bins[1]=='2018') & (np.isin(bins[2],['01','02','03','04','05','06','07','08','09'])) )]=6
bins2[np.where( (bins[1]=='2018') & (np.isin(bins[2],['10','11','12'])) | (bins[1]=='2019') & (np.isin(bins[2],['01','02','03','04','05','06','07','08','09'])) )]=7
bins2[np.where( (bins[1]=='2019') & (np.isin(bins[2],['10','11','12'])) | (bins[1]=='2020') & (np.isin(bins[2],['01','02','03','04','05','06','07','08','09'])) )]=8
bins2[np.where( (bins[1]=='2020') & (np.isin(bins[2],['10','11','12'])) | (bins[1]=='2021') & (np.isin(bins[2],['01','02','03','04','05','06','07','08','09'])) )]=9
bins2=np.vstack((bins[0],bins2))

bins=np.unique(bins2,axis=1,return_inverse=1)
binsum=np.bincount(bins[1],data)


export=np.vstack((bins[0][0],bins[0][1],binsum)).T

t2lookup=np.searchsorted(t2dict[0].astype(int),export[:,0].astype(int))
export[:,0]=t2dict[1][t2lookup]
period=np.array((np.unique(export[:,1]),['DISCARDED','SALES13','SALES14','SALES15','SALES16','SALES17','SALES18','SALES19','SALES20','SALES21']))
ttmlookup=np.searchsorted(period[0].astype(int),export[:,1].astype(int))
export[:,1]=period[1][ttmlookup]

rows=np.unique(t2)
rows=np.array(list(enumerate(rows.tolist())))
rownames=np.searchsorted(t2dict[0].astype(int),rows[:,1])
columns=period[1]
columns=np.array(list(enumerate(columns.tolist())))
columnnames=np.insert(columns[:,1],0,'TRANSLATION')

report=np.zeros((rows.shape[0],columns.shape[0]))
rowlookup=np.searchsorted(rows[:,1],export[:,0].astype(int))
columnlookup=np.searchsorted(columns[:,0].astype(int),export[:,1].astype(int))
report[rowlookup,columnlookup]=export[:,2]

#ADD ROWNAMES AND COLUMN NAMES
report=np.hstack((t2dict[1][rownames].reshape(-1,1),report))
report=np.delete(report,1,1)
keep=np.insert(np.where(report[1:,9].astype(float)>=10000)[0]+1,0,0)
report=report[keep]
report=report[(-report[:,9].astype(float)).argsort()]
report=np.vstack((np.delete(columnnames,1),report))
np.savetxt('/home/dmf/Documents/x.csv',report,fmt=('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s'))

