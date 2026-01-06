import numpy as np
import csv
import os
from dateconvert import convertdate
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import datetime
import numpy.lib.recfunctions as rf
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy.lib.recfunctions as rf
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

os.system('cp /home/dmf/Downloads/DEMAND_ORDERS_RPT.csv /home/dmf/csv/DEMAND_ORDERS_RPT.csv')
 
demand = []

with open("/home/dmf/csv/DEMAND_ORDERS_RPT.csv", "r") as f:
	contents = csv.reader(f)
	for c in contents:
		demand.append(c)

fields=['date','ordersty','ordersly','salesty','salesly']
for f in fields:
    vars()[f]=np.zeros(0)

for d in demand:
    date=np.append(date,d[0])
    ordersty=np.append(ordersty,d[1])
    ordersly=np.append(ordersly,d[4])
    salesty=np.append(salesty,d[5])
    salesly=np.append(salesly,d[8])

indices=np.isin(date,['DATE','WTD','MTD','YTD'],invert=True)
for f in fields:
    vars()[f]=vars()[f][indices]

for f in fields:
    vars()[f]=[ele.replace(",",'').replace("$","") for ele in vars()[f]]

date=[convertdate(ele) for ele in date]
ordersty=[int(ele) for ele in ordersty]
ordersly=[int(ele) for ele in ordersly]
salesty=[float(ele) for ele in salesty]
salesly=[float(ele) for ele in salesly]

arr1=list(zip(date,ordersty,ordersly,salesty,salesly))

dt=np.dtype([ ('date','M8[D]'), ('orders_ty','i4'), ('orders_ly','i4'), ('sales_ty','f4'), ('sales_ly','f4')  ])
arr1=np.array(arr1,dtype=dt)

week=np.asarray([x.astype(datetime.datetime).isocalendar()[1] for x in arr1['date'].astype('datetime64[D]')])
weekday=np.asarray([x.astype(datetime.datetime).isocalendar()[2] for x in arr1['date'].astype('datetime64[D]')])
year=arr1['date'].astype('datetime64[D]').astype('datetime64[Y]').astype(int)
quarter=np.zeros(len(arr1['date']),dtype=int)
quarter[ (arr1['date']>=np.datetime64('2022-01-01')) & (arr1['date']<=np.datetime64('2022-03-31'))]=1
quarter[ (arr1['date']>=np.datetime64('2022-04-01')) & (arr1['date']<=np.datetime64('2022-06-30'))]=2
quarter[ (arr1['date']>=np.datetime64('2022-07-01')) & (arr1['date']<=np.datetime64('2022-09-30'))]=3
quarter[ (arr1['date']>=np.datetime64('2022-10-01')) & (arr1['date']<=np.datetime64('2022-12-31'))]=4

orders_diff=arr1['orders_ty']-arr1['orders_ly']
sales_diff=arr1['sales_ty'].astype('int32')-arr1['sales_ly'].astype('int32')
sales_diff2=sales_diff/100
xaxis=np.arange(len(arr1['date']))
z0 = np.polyfit(xaxis, orders_diff, 1)
p0 = np.poly1d(z0)
z1 = np.polyfit(xaxis, sales_diff2, 1)
p1 = np.poly1d(z1)

#animated gif scatterplot 
colors=np.array(['black','red','green','blue','orange'])
zipcoords=np.hstack((orders_diff.reshape(-1,1),sales_diff.reshape(-1,1)))

fig, ax = plt.subplots(figsize=(10, 10))
scat = ax.scatter(x=orders_diff,
                  y=sales_diff,
                  c=colors[quarter])

def animationUpdate(k):
    i = np.arange(len(zipcoords))[:k]
    scat.set_offsets(zipcoords[i])
    return scat,

def init():
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('orders', size=14, labelpad=-24, x=1.03)
    ax.set_ylabel('sales', size=14, labelpad=-21, y=1.02, rotation=0)
    ax.ticklabel_format(style='plain')
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    custom_lines = [Line2D([0], [0], color='red', lw=4),Line2D([0], [0], color='green', lw=4),Line2D([0], [0], color='blue', lw=4),Line2D([0], [0], color='orange', lw=4)]
    ax.legend(custom_lines,['Q1','Q2','Q3','Q4'],loc='lower right')


anim = animation.FuncAnimation(fig,animationUpdate,init_func=init,interval=20,repeat=False,frames=len(xaxis))
fig.suptitle('X axis: Y/Y Orders Diff, Y Axis: Y/Y Sales Diff', fontsize=14)
with open("/home/dmf/Videos/scatter.html", "w") as f:
    print(anim.to_html5_video(), file=f)

plt.close()

#animated line charts
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (15,5))
axes2 = axes.twinx()
axes.axhline(c='red')

axes.set_xlim(np.datetime64('2022-01-01'), np.max(arr1['date']).astype('datetime64[D]'))
axes.set_ylim(-12000,12000)
axes2.set_ylim(-12000,12000)
axes.tick_params(axis ='y', labelcolor = 'blue')
axes2.tick_params(axis ='y', labelcolor = 'green')

#add x-axis label
axes.set_xlabel('Date', fontsize=14)

#add y-axis label
axes.set_ylabel('Orders', color='blue', fontsize=16)

#add second y-axis label
axes2.set_ylabel('Sales (hundreds)', color='green', fontsize=16)

#plt.style.use("ggplot")

x0,y0,y1,z0,z1 = [], [], [], [], []

def animate(i):
    x0.append((arr1['date'].astype('datetime64[D]')[i]))
    y0.append((orders_diff[i]))
    y1.append((sales_diff2[i]))
    z0.append((p0(xaxis)[i]))
    z1.append((p1(xaxis)[i]))
    axes.plot(x0,y0, color='blue')
    axes.plot(x0,z0,color='blue',linestyle='dashed')
    axes2.plot(x0,y1,color='green')
    axes2.plot(x0,z1,color='green',linestyle='dashed')

anim = animation.FuncAnimation(fig, animate, interval=20, repeat=False,frames=len(xaxis))
fig.suptitle('Y/Y Orders Diff, Y/Y Sales Diff, Dashed Trendlines', fontsize=14)
with open("/home/dmf/Videos/line.html", "w") as f:
    print(anim.to_html5_video(), file=f)

plt.close()

