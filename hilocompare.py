import numpy as np

fname1='high2017_2022'
fname2='low2017_2022'
high=np.load('/home/dmf/python/npz/'+fname1+'.npz')['arr_0']
low=np.load('/home/dmf/python/npz/'+fname2+'.npz')['arr_0']
high=np.sort(high,order=('ORDER_NO','SKU'))
low=np.sort(low,order=('ORDER_NO','SKU'))

print(len(high),len(low),len(high)-len(low))
np.array_equal(high['ORDER_NO'],low['ORDER_NO'])
np.array_equal(high['SKU'],low['SKU'])

#RECONCILIATION
#each file contains the same order numbers
statsbefore=np.array([len(high),len(low)])
keep=np.intersect1d(high['ORDER_NO'],low['ORDER_NO'])
high=high[np.isin(high['ORDER_NO'],keep)]
low=low[np.isin(low['ORDER_NO'],keep)]
statsafter=np.array([len(high),len(low)])
print(statsbefore,statsafter,statsafter-statsbefore)

#each file contains orders with the same number of lines
statsbefore=np.array([len(high),len(low)])
print(len(high),len(low))
x=np.unique(high['ORDER_NO'],return_counts=True)
y=np.unique(low['ORDER_NO'],return_counts=True)
np.array_equal(x[0],y[0])
idx=np.where(x[1]==y[1])
keep=x[0][idx]
high=high[np.isin(high['ORDER_NO'],keep)]
low=low[np.isin(low['ORDER_NO'],keep)]
statsafter=np.array([len(high),len(low)])
print(statsbefore,statsafter,statsafter-statsbefore)

#confirm high low files match line for line
high=np.sort(high,order=('ORDER_NO','SKU'))
low=np.sort(low,order=('ORDER_NO','SKU'))
np.array_equal(high['ORDER_NO'],low['ORDER_NO'])
np.array_equal(high['SKU'],low['SKU'])

#save high low files
np.savez_compressed('/home/dmf/python/npz/'+fname1+'.npz',high)
np.savez_compressed('/home/dmf/python/npz/'+fname2+'.npz',low)

#TROUBLESHOOTING
np.setdiff1d(high['ORDER_NO'],low['ORDER_NO'])
np.setdiff1d(low['ORDER_NO'],high['ORDER_NO'])

for i in np.arange(len(high['ORDER_NO'])):
    if high['ORDER_NO'][i] != low['ORDER_NO'][i]:
        print(high['ORDER_NO'][i])
        print(low['ORDER_NO'][i])
        print(high['ORDER_NO'][i-5:i+5])
        print(low['ORDER_NO'][i-5:i+5])
        break

for i in np.arange(len(high['SKU'])):
    if high['SKU'][i] != low['SKU'][i]:
        print(high['ORDER_NO'][i])
        print(low['ORDER_NO'][i])
        print(high['ORDER_NO'][i-5:i+5])
        print(low['ORDER_NO'][i-5:i+5])
        break

idx=127489502
high=high[high['ORDER_NO']!=idx]
low=low[low['ORDER_NO']!=idx]


