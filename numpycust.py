import numpy as np
from numpy import genfromtxt
from datetime import datetime 
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

def rec(x):
    return (datetime.now()-x).astype('timedelta64[D]')

str2date = lambda x: datetime.strptime(x, '%m/%d/%Y')
str2year = lambda x: datetime.strptime(x, '%m/%d/%Y').year
str2month = lambda x: datetime.strptime(x, '%m/%d/%Y').month
str2day = lambda x: datetime.strptime(x, '%m/%d/%Y').day
t1 = lambda x: x[0:2]

#IMPORT 1D ARRAYS
acid=genfromtxt('/home/dmf/csv/skucodedetails2021.txt',dtype='int64',usecols=(0),names=('CID'),skip_header=1,delimiter='\t',encoding='latin1',converters={0: lambda s: int(s or 0)})

asku=genfromtxt('/home/dmf/csv/skucodedetails2021.txt',dtype='str',usecols=(4),names=('SKU'),skip_header=1,delimiter='\t',encoding='latin1',converters={4: lambda s: str(s or '')})

aoid=genfromtxt('/home/dmf/csv/skucodedetails2021.txt',dtype='int64',usecols=(1),names=('OID'),skip_header=1,delimiter='\t',encoding='latin1',converters={1: lambda s: int(s or 0)})

adate=genfromtxt('/home/dmf/csv/skucodedetails2021.txt',dtype='str',usecols=(2),names=('DATE'),skip_header=1,delimiter='\t',encoding='latin1',converters={2: lambda s: str2date(s)})

aprice=genfromtxt('/home/dmf/csv/skucodedetails2021.txt',dtype='float64',usecols=(6),names=('PRICE'),skip_header=1,delimiter='\t',encoding='latin1',converters={6: lambda s: float(s or 0)})

aunits=genfromtxt('/home/dmf/csv/skucodedetails2021.txt',dtype='int64',usecols=(7),names=('UNITS'),skip_header=1,delimiter='\t',encoding='latin1',missing_values=0,filling_values=0,converters={7: lambda s: int(s or 0)})
asales=aunits*aprice

cidsort=np.argsort(acid)

#RECENCY
dates=rec(adate)
dates=dates.astype(int)

recency=[]
recencybycid=np.split(dates[cidsort], np.unique(acid[cidsort], return_index=True)[1][1:])

for i in enumerate(recencybycid):
    recency.append (np.min(i[1]))

recency=np.array(recency)

#FREQUENCY
frequency=[]
freqbycid=np.split(aoid[cidsort], np.unique(acid[cidsort], return_index=True)[1][1:])

for i in enumerate(freqbycid):
    frequency.append (len(np.unique(i[1])))

frequency=np.array(frequency)

#MONETARY
monetary=[]
monbycid=np.split(asales[cidsort], np.unique(acid[cidsort], return_index=True)[1][1:])
for i in enumerate(salesbycid):
    monetary.append (np.sum(i[1]))

monetary=np.array(monetary)

#UNITS
units=[]
unitsbycid=np.split(aunits[cidsort], np.unique(acid[cidsort], return_index=True)[1][1:])

for i in enumerate(unitsbycid):
    units.append (np.sum(i[1]))

units=np.array(units)

#SKUS
skus=[]
skusbycid=np.split(asku[cidsort], np.unique(acid[cidsort], return_index=True)[1][1:])

for i in enumerate(skusbycid):
    skus.append (len(np.unique(i[1])))

skus=np.array(skus)

#PARTS
aparts=genfromtxt('/home/dmf/csv/parts.txt',skip_header=1,delimiter=',',dtype='str',encoding='latin1',usecols=(6,11,12),converters={6: lambda s: str(s or ''),11: lambda s: str(s or ''),12: lambda s: str(s or '')})
aparts=aparts[aparts[:,0].argsort()]
askulookup=np.searchsorted(aparts[:,0],asku)

#CATEGORIES
acat=aparts[askulookup,1]

t1=[]
for i in enumerate(acat):
    t1.append (i[1][0:2])

t1=np.array(t1)

t2=[]
for i in enumerate(acat):
    t2.append (i[1][3:6])

t2=np.array(t2)

t3=[]
for i in enumerate(acat):
    t3.append (i[1][7:])

t3=np.array(t3)

abg=aparts[askulookup,2]

#TIER2
tier2=[]
tier2bycid=np.split(t2[cidsort], np.unique(acid[cidsort], return_index=True)[1][1:])
for i in enumerate(tier2bycid):
    tier2.append (len(np.unique(i[1]))) 

tier2=np.array(tier2)


#RFMUSC
rfmust=np.array((recency,frequency,monetary,units,skus,tier2))
rfmust.T

