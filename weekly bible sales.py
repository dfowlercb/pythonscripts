#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
from scipy import sparse
import os
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# In[6]:


np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})


# In[7]:


#LOAD FILES
os.chdir('/home/dmf/python/npz')


# In[8]:


#order level   
arrays=np.load('oldechilo.npz',allow_pickle=1)
oidol=arrays['oidol']
date=arrays['date']


# In[9]:


#line level
arrays=np.load('lldechilo.npz',allow_pickle=1)
oid=arrays['oid']
sku=arrays['sku']
units=arrays['units']
price=arrays['price']
parts=arrays['parts']
ll_link=arrays['ll_link']


# In[10]:


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


# In[11]:


#find bibles 
bibles=np.where(parts[:,3]=='02')
i=np.isin(sku,bibles)


# In[12]:


#matrices
data=units[i]*price[i]


# In[13]:


#annual bible sales
row=np.repeat(0,data.size)
col=years[1][ll_link][i]
A=sparse.coo_matrix((data,(row,col)))
A.toarray()


# In[14]:


#weekly bible sales
row=weeks[1][ll_link][i]
row[np.where(row>51)[0]]=51
col=years[1][ll_link][i]
A=sparse.coo_matrix((data,(row,col)))
A.toarray()

os.chdir('/home/dmf/Documents')
np.savetxt('bibles.txt',A.toarray(),delimiter=',',fmt='%f')


# In[15]:


#LINE CHART
x=range(1,53)
y2021=np.transpose(A.toarray())[24]
y2020=np.transpose(A.toarray())[23]
y2019=np.transpose(A.toarray())[22]
y2018=np.transpose(A.toarray())[21]
y2017=np.transpose(A.toarray())[20]


# In[77]:


fig, ax = plt.subplots()
y_2017, = ax.plot(y2017, label='2018',linewidth=1)
y_2018, = ax.plot(y2018, label='2019',linewidth=1)
y_2019, = ax.plot(y2019, label='2020',linewidth=1)
y_2020, = ax.plot(y2020, label='2020',linewidth=1)
y_2021, = ax.plot(y2021, label='2021',linewidth=1)

ax.legend([y_2017, y_2018, y_2019, y_2020, y_2021], ['2017', '2018', '2019','2020', '2021'],loc='best', bbox_to_anchor=(.75,.93,.25,.25),ncol=5,fancybox=True, shadow=True)
ax.set_title('Weekly Bible Sales',fontsize=8)
ax.set_xlabel('WEEK #')
ax.set_ylabel('MERCH $')
ax.set_xlim(0, 52)
ax.set_ylim(0, 2225000)
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.ticklabel_format(style='plain',axis='both')
ax.grid(which='major')
ax.grid(which='minor',ls='--')
plt.savefig('chart.pdf',facecolor='w')


# In[71]:


help(ax.set_title)


# In[ ]:




