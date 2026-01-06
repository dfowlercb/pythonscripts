import numpy as np
import csv
import os
from pyfunc/dateconvert import convertdate
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import datetime
import numpy.lib.recfunctions as rf
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy.lib.recfunctions as rf
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

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
quarter[ (arr1['date']>=np.datetime64('2024-01-01')) & (arr1['date']<=np.datetime64('2022-03-31'))]=1
quarter[ (arr1['date']>=np.datetime64('2024-04-01')) & (arr1['date']<=np.datetime64('2022-06-30'))]=2
quarter[ (arr1['date']>=np.datetime64('2024-07-01')) & (arr1['date']<=np.datetime64('2022-09-30'))]=3
quarter[ (arr1['date']>=np.datetime64('2024-10-01')) & (arr1['date']<=np.datetime64('2022-12-31'))]=4

orders_diff=arr1['orders_ty']-arr1['orders_ly']
sales_diff=arr1['sales_ty'].astype('int32')-arr1['sales_ly'].astype('int32')
z0 = np.polyfit(np.arange(len(orders_diff)), orders_diff, 1)
p0 = np.poly1d(z0)

#static scatterplot
colors=np.array([ 'black','red','green','blue','yellow','darkgreen','orange','cyan' ])[quarter]
size=np.array([0,2,4,6,8])[quarter]
markers=np.array(['.',',','o','v'])[quarter]

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x=orders_diff, y=sales_diff,c=colors,s=size,marker=markers)
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('orders', size=14, labelpad=-24, x=1.03)
ax.set_ylabel('sales', size=14, labelpad=-21, y=1.02, rotation=0)
ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color='red',  lw=4),
                Line2D([0], [0], color='green', lw=4),
                Line2D([0], [0], color='blue', lw=4)]
ax.legend(custom_lines,['Quarter1','Quarter2','Quarter3'])
plt.show()


#animated gif scatterplot 
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
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    custom_lines = [Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='green', lw=4),
                Line2D([0], [0], color='blue', lw=4)]
    ax.legend(custom_lines,['Quarter1','Quarter2','Quarter3'],loc='lower right')


animation = FuncAnimation(fig,animationUpdate,init_func=init,interval=1,save_count=500)
writer = PillowWriter(fps=20)  
animation.save('/home/dfowler/documents/sales_orders.gif', writer=writer)


#animated line charts
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (15,5))
axes2 = axes.twinx()
axes.set_xlim(np.datetime64('2022-01-01'), np.max(arr1['date']).astype('datetime64[D]'))
axes.set_ylim(np.min(orders_diff), np.max(orders_diff))
axes2.set_ylim(np.min(sales_diff), np.max(sales_diff))

#add x-axis label
axes.set_xlabel('Date', fontsize=14)

#add y-axis label
axes.set_ylabel('Orders', color='blue', fontsize=16)

#add second y-axis label
axes2.set_ylabel('Sales', color='green', fontsize=16)

plt.style.use("ggplot")

x1,y1,y2,z1 = [], [], [], []

def animate(i):
    x1.append((arr1['date'].astype('datetime64[D]')[i]))
    y1.append((orders_diff[i]))
    y2.append((sales_diff[i]))
    z1.append((p0(orders_diff)[i]))
    axes.plot(x1,y1, color='blue')
    axes.plot(x1,z1,color='gray')
    axes2.plot(x1,y2,color='green')

animation = FuncAnimation(fig, animate, interval=1, save_count=229)
writer = PillowWriter(fps=20)  
animation.save('/home/dmf/documents/sales_orders_line.gif', writer=writer)



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

