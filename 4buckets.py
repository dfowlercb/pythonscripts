import pandas as pd
import numpy as np

o16=pd.read_csv('~/csv/skucodedetails2016.txt',sep='\t',encoding='latin-1',usecols=['CM_ID','ORDER_NO','ORDER_DATE','SKU','UNIT_PRICE','UNITS','ORDER_TYPE','HOUSEHOLD ','LINE STATUS','BILLTO STATE','BILLTO_ZIPCODE','UNIT_COST'],dtype={'CM_ID':str,'ORDER_NO':str,'HOUSEHOLD ':str,'BILLTO_ZIPCODE':str})
o17=pd.read_csv('~/csv/skucodedetails2017.txt',sep='\t',encoding='latin-1',usecols=['CM_ID','ORDER_NO','ORDER_DATE','SKU','UNIT_PRICE','UNITS','ORDER_TYPE','HOUSEHOLD ','LINE STATUS','BILLTO STATE','BILLTO_ZIPCODE','UNIT_COST'],dtype={'CM_ID':str,'ORDER_NO':str,'HOUSEHOLD ':str,'BILLTO_ZIPCODE':str})
o18=pd.read_csv('~/csv/skucodedetails2018.txt',sep='\t',encoding='latin-1',usecols=['CM_ID','ORDER_NO','ORDER_DATE','SKU','UNIT_PRICE','UNITS','ORDER_TYPE','HOUSEHOLD ','LINE STATUS','BILLTO STATE','BILLTO_ZIPCODE','UNIT_COST'],dtype={'CM_ID':str,'ORDER_NO':str,'HOUSEHOLD ':str,'BILLTO_ZIPCODE':str})
o19=pd.read_csv('~/csv/skucodedetails2019.txt',sep='\t',encoding='latin-1',usecols=['CM_ID','ORDER_NO','ORDER_DATE','SKU','UNIT_PRICE','UNITS','ORDER_TYPE','HOUSEHOLD ','LINE STATUS','BILLTO STATE','BILLTO_ZIPCODE','UNIT_COST'],dtype={'CM_ID':str,'ORDER_NO':str,'HOUSEHOLD ':str,'BILLTO_ZIPCODE':str})
o20=pd.read_csv('~/csv/skucodedetails2020.txt',sep='\t',encoding='latin-1',usecols=['CM_ID','ORDER_NO','ORDER_DATE','SKU','UNIT_PRICE','UNITS','ORDER_TYPE','HOUSEHOLD ','LINE STATUS','BILLTO STATE','BILLTO_ZIPCODE','UNIT_COST'],dtype={'CM_ID':str,'ORDER_NO':str,'HOUSEHOLD ':str,'BILLTO_ZIPCODE':str})
o21=pd.read_csv('~/csv/skucodedetails2021.txt',sep='\t',encoding='latin-1',usecols=['CM_ID','ORDER_NO','ORDER_DATE','SKU','UNIT_PRICE','UNITS','ORDER_TYPE','HOUSEHOLD ','LINE STATUS','BILLTO STATE','BILLTO_ZIPCODE','UNIT_COST'],dtype={'CM_ID':str,'ORDER_NO':str,'HOUSEHOLD ':str,'BILLTO_ZIPCODE':str})
df=pd.concat([o16,o17,o18,o19,o20,o21])
del [o16,o17,o18,o19,o20,o21]
df=df.fillna({'SKU':'','UNITS':0,'ORDER_TYPE':'W','UNIT_COST':0,'BILLTO STATE':''})
df.isnull().sum()
df.info()

df=df[~(df['LINE STATUS']=='cancelled')]
df=df[df.ORDER_TYPE.isin(['W','P','M'])]
df=df.drop(columns=['ORDER_TYPE','LINE STATUS'])

df['ORDER_DATE']=pd.to_datetime(df['ORDER_DATE'])
df['YEAR']=pd.DatetimeIndex(df['ORDER_DATE']).year
df['SALES']=df.UNITS*df.UNIT_PRICE
df['COST']=df['UNITS']*df['UNIT_COST']
df['MARGIN']=df['SALES']-df['COST']
df['BUCKET']=0

df=df.set_index('SKU',drop=False)
parts=pd.read_csv('~/csv/parts.txt',sep=',',encoding='latin-1',usecols=['part','category','BUYER_GROUP'])
parts=parts.loc[parts['part'].isin(df.index)]
parts=parts.drop_duplicates(subset='part')
parts=parts.set_index('part')
df=df.join(parts,how='left')
df['T1']=df['category'].str.slice(0,2)
df['T2']=df['category'].str.slice(3,6)
df['T3']=df['category'].str.slice(7)
df.to_pickle('~/python/pkl/consensus.pkl')

#CREATE MULTIDIMENSIONAL ARRAY
o21['CM_ID']=o21['CM_ID'].astype('int64')
o21['ORDER_NO']=o21['ORDER_NO'].astype('int64')
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

cust=len(np.unique(o21['CM_ID'].to_numpy()))
rfm=np.zeros((5,cust,1))

x=o21[['CM_ID','ORDER_NO','UNITS','UNIT_PRICE']].to_numpy()
freq=np.unique(x[:,0:2],axis=0)
unique, counts = np.unique(freq[:,0], return_counts=True)
freq=np.asarray((unique,counts)).T
freq.shape

####################################
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.2f}'.format

df=pd.read_pickle('~/python/pkl/consensus.pkl')
df['CM_ID']=df['HOUSEHOLD ']
df.value_counts('BUCKET')

#HOMESCHOOL BUCKET      
 #HMS 50%
hms=df.loc[df['T2']=='HMS'].groupby('CM_ID').agg({'SALES':'sum'})
sales=df.loc[df['CM_ID'].isin(hms.index)].groupby('CM_ID').agg({'SALES':'sum'})
hms=hms.join(sales,rsuffix='all')
hms=hms.query('SALES/SALESall>.50')
df.loc[df['CM_ID'].isin(hms.index),'BUCKET']=1
df.value_counts('BUCKET')
 #HMS TOP
bg=df.pivot_table(index='CM_ID',columns='BUYER_GROUP',values='SALES',fill_value=0)
bgmax=bg.idxmax(axis=1)
df.loc[df['CM_ID'].isin(bgmax[bgmax=='HOMESCHOOLING'].index),'BUCKET']=1
df.value_counts('BUCKET')

#CHURCH BUCKET
 #CHURCH FLAG / TAX EXEMPT FLAG
flag=pd.read_csv('~/R/projects/consensus/churchconsensus.txt',dtype=str)
df.loc[(df['BUCKET']==0)&(df['CM_ID'].isin(flag['HOUSEHOLD ID'])),['BUCKET']]=2
 #BIB
bib=pd.read_csv('~/R/projects/consensus/bib.txt',sep='\t',usecols=['CM_ID','ORDER_NO','ORDER_DATE','SKU','UNIT_PRICE','UNITS','HOUSEHOLD ','LINE STATUS','BILLTO STATE','BILLTO_ZIPCODE','UNIT_COST'])
bib=bib.fillna({'SKU':'','UNITS':0,'UNIT_COST':0,'NEW_CUST.FLAG':'','BILLTO STATE':''})
bib.isnull().sum()
bib.info()
bib=bib.loc[bib['LINE STATUS'].isin(['backordered','shipped'])]
bib=bib.drop(columns='LINE STATUS')
bib['ORDER_DATE']=pd.to_datetime(bib['ORDER_DATE'])
bib['YEAR']=pd.DatetimeIndex(bib['ORDER_DATE']).year
bib['SALES']=bib['UNITS']*bib['UNIT_PRICE']
bib['COST']=bib['UNITS']*bib['UNIT_COST']
bib['MARGIN']=bib['SALES']-bib['COST']
bib['BUCKET']=2
parts=pd.read_csv('~/csv/parts.txt',sep=',',encoding='latin-1',usecols=['part','category','BUYER_GROUP'])
parts=parts.set_index('part')
bib=bib.set_index('SKU',drop=False)
bib=bib.join(parts,how='left')
del(parts)
bib['T1']=bib['category'].str.slice(0,2)
bib['T2']=bib['category'].str.slice(3,6)
bib['T3']=bib['category'].str.slice(7)
bib['CM_ID']=bib['HOUSEHOLD ']
bib['CM_ID']=bib['CM_ID'].apply(lambda x: f'b{x}')
bib['ORDER_NO']=bib['ORDER_NO'].apply(lambda x: f'b{x}')
bib['HOUSEHOLD ']=bib['HOUSEHOLD '].apply(lambda x: f'b{x}')
df=pd.concat([df,bib])

#BUCKET 3 PROFESIONAL
pro=df.loc[(df['BUCKET']==0)&((df['T2'].isin(['APO','BBS','BRF','BST','CHH','CHR','CMM','CSL','CUR','DSS','EVA','GNR','INS','LAN','LEA','PHI','PRE','PSY','THE','VBS']))|(df['T3'].isin(['GA','PP','EM','BP','LD','YM','PR'])))]
pro=pro.groupby('CM_ID').agg({'SALES':sum})
pro=pro.query('SALES>=100')
df.loc[df['CM_ID'].isin(pro.index),['BUCKET']]=3

#BUCKET 4 CONSUMER
df.loc[df['BUCKET']==0,['BUCKET']]=4

#SUMMARIZE BUCKETS AND SAVE TO PICKLE
df.groupby('BUCKET').agg({'SALES':sum})
df.to_pickle('~/python/pkl/consensus.pkl')


#LOAD BUCKETS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df=pd.read_pickle('~/python/pkl/consensus.pkl')

#NUMPY ARRAYS


#CHARTING

 #recency
x=df.groupby(['CM_ID','BUCKET'])['ORDER_DATE'].max()-pd.to_datetime('today')
x=x.dt.days.astype('int64').to_frame()
recency=x.pivot_table(index='BUCKET',values='ORDER_DATE',aggfunc=np.mean)
 #frequency
x=df.groupby(['CM_ID','BUCKET'])['ORDER_NO'].nunique().to_frame()
frequency=x.pivot_table(index='BUCKET',values='ORDER_NO',aggfunc=np.mean)
 #monetary
x=df.groupby(['CM_ID','BUCKET'])['SALES'].sum().to_frame()
monetary=x.pivot_table(index='BUCKET',values='SALES',aggfunc=np.mean)
rfm=pd.concat([recency,frequency,monetary],axis=1)

from sklearn import preprocessing


 #linear interpolation from -1 to +1
recency2=np.interp(np.array(recency), (np.array(recency).min(), np.array(recency).max()), (-1, 1))
frequency2=np.interp(np.array(frequency), (np.array(frequency).min(), np.array(frequency).max()), (-1, 1))
monetary2=np.interp(np.array(monetary), (np.array(monetary).min(), np.array(monetary).max()), (-1, 1))
#rfm=pd.concat([pd.DataFrame(recency2),pd.DataFrame(frequency2),pd.DataFrame(monetary2)],axis=1)


#PIE CHART
labels = 'HOMESCHOOL', 'CHURCH', 'PROFESSIONAL', 'CONSUMER'
sizes =df.loc[df['YEAR']==2020].groupby('BUCKET')['CM_ID'].nunique()/df.loc[df['YEAR']==2020,'CM_ID'].nunique()
sizes=sizes.to_list()
explode = (0, 0, 0, 0.1)  # only "explode" the 4th slice (i.e. 'CONSUMER')
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

#3D RFM CHART
 #3d plot without interpolation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs=rfm.iloc[:,0]
ys=rfm.iloc[:,1]
zs=rfm.iloc[:,2]

ax.scatter(xs=xs,ys=ys,zs=zs,c=['red','orange','tab:cyan','green'])
ax.text(xs[1],ys[1],zs[1],'HOMESCHOOL')
ax.text(xs[2],ys[2],zs[2],'CHURCH')
ax.text(xs[3],ys[3],zs[3],'PROFESSIONAL')
ax.text(xs[4],ys[4],zs[4],'CONSUMER')

ax.set_xlabel('recency')
ax.set_ylabel('frequency')
ax.set_zlabel('monetary')

rfm=rfm.round(1).astype(str)
list_2d=[ rfm.iloc[0].to_list() , rfm.iloc[1].to_list() , rfm.iloc[2].to_list(), rfm.iloc[3].to_list()] 
data_table = plt.table(	cellText=list_2d,rowLabels=('HOMESCHOOL','CHURCH','PROFESSIONAL','CONSUMER'),colLabels=('R','F','M'),rowColours=['red','orange','tab:cyan','green'],colWidths=[0.25]*4,loc='bottom left')
data_table.auto_set_font_size(False)
data_table.set_fontsize(16)
data_table.scale(.50,.60)
plt.title('FOUR BUSINESS UNITS: MEAN OF [RECENCY FREQUENCY MONETARY]',loc='center')
plt.show()


#POLAR MONETARY
data = np.array(rfm.SALES)
N = len(data)
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = data
width = 2 * np.pi / N

ax = plt.subplot(111, polar=True)
bars = ax.bar(theta, radii, width=width, bottom=0.0,color=['red','yellow','green','blue'])

ax.yaxis.set_ticks([200,400,600,800])
ax.xaxis.set_ticks(theta)
ax.xaxis.set_ticklabels(['HOMESCHOOL','CHURCH','PROFESSIONAL','CONSUMER'])
ax.legend()
plt.show()

#POLAR RECENCY
data = np.array(rfm.ORDER_DATE)
N = len(data)
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = abs(data)

ax = plt.subplot(111, polar=True)
dots = ax.scatter(theta, radii,color=['red','yellow','green','blue'])

ax.set_ylim(500,900 )
ax.yaxis.set_ticks([500,600,700,800,900])
ax.xaxis.set_ticks(theta)
ax.xaxis.set_ticklabels(['HOMESCHOOL','CHURCH','PROFESSIONAL','CONSUMER'])
ax.legend()
plt.show()

#CARTESIAN TO POLAR 
import math
from operator import add
from operator import truediv

def power(my_list):
    return [ x**2 for x in my_list ]

def square(my_list):
    return [math.sqrt(x) for x in my_list]

def arctan (my_list1):
        return list(map(np.arctan,my_list1))

x=np.array(recency)
y=np.array(frequency)

#GET R FOR  X,Y  
x2plusy2=list(map(add,power(x),power(y)))
r=square(x2plusy2)

#GET THETA FOR X,Y
ydivx=list(map(truediv,y,x))
theta=arctan(ydivx)

#add pi when x<0
for i, item in enumerate(x):
	if item < 0:
		theta[i] = theta[i]+np.pi;

#create x labelsx
xlabels=[]

for i in monetary['SALES'].to_list():
            xlabels.append(str(round(i)));

size=np.interp(np.array(monetary), (np.array(monetary).min(), np.array(monetary).max()), (40, 160))

#POLAR GRAPH
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
#ax.set_ylim(bottom=min(r),top=max(r)+.2)
ax.scatter(recency2,frequency2,c=['red','green','yellow','blue'])
plt.show()

#CARTESIAN SALES & ORDERS
import datetime as dt
dfaug20=df.loc[df['ORDER_DATE'].between('2020-08-02','2020-09-01')]
dfaug21=df.loc[df['ORDER_DATE'].between('2021-08-01','2021-08-31')]

x0=dfaug20.groupby(['ORDER_DATE','BUCKET'])['ORDER_NO'].nunique().reset_index()
x1=dfaug21.groupby(['ORDER_DATE','BUCKET'])['ORDER_NO'].nunique().reset_index()
x0=x0.set_index(['ORDER_DATE','BUCKET'])
x1=x1.set_index(['ORDER_DATE','BUCKET'])
x0.index=x1.index
x=x0.join(x1,lsuffix='_LY',rsuffix='_TY')
x['ORDERS_DIFF']=x['ORDER_NO_TY']-x['ORDER_NO_LY']

y0=dfaug20.groupby(['ORDER_DATE','BUCKET'])['SALES'].sum().reset_index()
y1=dfaug21.groupby(['ORDER_DATE','BUCKET'])['SALES'].sum().reset_index()
y0=y0.set_index(['ORDER_DATE','BUCKET'])
y1=y1.set_index(['ORDER_DATE','BUCKET'])
y0.index=y1.index
y=y0.join(y1,lsuffix='_LY',rsuffix='_TY')
y['SALES_DIFF']=y['SALES_TY']-y['SALES_LY']

cart=x.join(y)
cart=cart.reset_index(drop=False)
colors={1:'Red',2:'Green',3:'Yellow',4:'Blue'}

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x=cart['ORDERS_DIFF'].to_list(), y=cart['SALES_DIFF'].to_list(),c=cart['BUCKET'].map(colors))
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('orders', size=14, labelpad=-24, x=1.03)
ax.set_ylabel('sales', size=14, labelpad=-21, y=1.02, rotation=0)
ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='green', lw=4),
                Line2D([0], [0], color='yellow', lw=4),
                Line2D([0], [0], color='blue', lw=4)]
ax.legend(custom_lines,['HOMESCHOOL','CHURCH','PROFESSIONAL','CONSUMER'])
plt.show()


#animated gif version
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:0.0f}'.format})
segment={1:'HOMESCHOOL',2:'CHURCH',3:'PROFESSIONAL',4:'CONSUMER'}
colors={1:'red',2:'green',3:'yellow',4:'blue'}
cart['color'] = cart['BUCKET'].map(colors)
cart['segment']=cart['BUCKET'].map(segment)

sales_orders = np.zeros(shape=(len(cart)), dtype=[('position', float, (2,)),
                                          ('color',int)])

sales_orders['position']=list(zip(cart['ORDERS_DIFF'],cart['SALES_DIFF']))
sales_orders['color']=cart['BUCKET'].to_numpy()

fig, ax = plt.subplots(figsize=(10, 10))
scat = ax.scatter(x=sales_orders['position'][:, 0],
                  y=sales_orders['position'][:, 1],
                  c=pd.DataFrame(sales_orders['color'],
                  columns=['color'])['color'].map(colors))

def animationUpdate(k):
    i = list(range(len(sales_orders)))[:k]
    scat.set_offsets(sales_orders['position'][i])
    return scat,

def init():
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('orders', size=14, labelpad=-24, x=1.03)
    ax.set_ylabel('sales', size=14, labelpad=-21, y=1.02, rotation=0)
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    custom_lines = [Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='green', lw=4),
                Line2D([0], [0], color='yellow', lw=4),
                Line2D([0], [0], color='blue', lw=4)]
    ax.legend(custom_lines,['HOMESCHOOL','CHURCH','PROFESSIONAL','CONSUMER'],loc='lower right')


animation = FuncAnimation(fig,animationUpdate,init_func=init,interval=10000)
writer = PillowWriter(fps=1)  
animation.save('sales_orders.gif', writer=writer)


#CARTESIAN FOUR COORDINATES examples
 #example 1
# Enter x and y coordinates of points and colors
xs = [0, 2, -3, -1.5]
ys = [0, 3, 1, -2.5]
colors = ['m', 'g', 'r', 'b']


# Plot points
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(xs, ys, c=colors)

# Draw lines connecting points to axes
for x, y, c in zip(xs, ys, colors):
    ax.plot([x, x], [0, y], c=c, ls='--', lw=1.5, alpha=0.5)
    ax.plot([0, x], [y, y], c=c, ls='--', lw=1.5, alpha=0.5)

# Set identical scales for both axes
ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')

# Set bottom and left spines as x and y axes of coordinate system
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Create 'x' and 'y' labels placed at the end of the axes
ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)

# Create custom major ticks to determine position of tick labels
x_ticks = np.arange(xmin, xmax+1, ticks_frequency)
y_ticks = np.arange(ymin, ymax+1, ticks_frequency)
ax.set_xticks(x_ticks[x_ticks != 0])
ax.set_yticks(y_ticks[y_ticks != 0])

# Create minor ticks placed at each integer to enable drawing of minor grid
# lines: note that this has no effect in this example with ticks_frequency=1
ax.set_xticks(np.arange(xmin, xmax+1), minor=True)
ax.set_yticks(np.arange(ymin, ymax+1), minor=True)

# Draw major and minor grid lines
ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

# Draw arrows
arrow_fmt = dict(markersize=4, color='black', clip_on=False)
ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)

plt.show()

 #example 2
xmin, xmax, ymin, ymax = -5, 5, -5, 5
ticks_frequency = 1
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('#ffffff')
ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('$x$', size=14, labelpad=-24, x=1.02)
ax.set_ylabel('$y$', size=14, labelpad=-21, y=1.02, rotation=0)
plt.text(0.49, 0.49, r"$O$", ha='right', va='top',transform=ax.transAxes,horizontalalignment='center', fontsize=14)
x_ticks = np.arange(xmin, xmax+1, ticks_frequency)
y_ticks = np.arange(ymin, ymax+1, ticks_frequency)
ax.set_xticks(x_ticks[x_ticks != 0])
ax.set_yticks(y_ticks[y_ticks != 0])
ax.set_xticks(np.arange(xmin, xmax+1), minor=True)
ax.set_yticks(np.arange(ymin, ymax+1), minor=True)
ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

x = np.linspace(-5, 10, 100)
y = func(x)

plt.plot(x, y, 'b', linewidth=2)
plt.show()
