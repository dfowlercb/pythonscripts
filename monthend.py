import numpy as np
monthend=np.genfromtxt('/home/dmf/csv/month_end_dates.csv',dtype='int',names=True)
def monthendconvert(shipdate):
    for x in np.arange(len(monthend)):    
        idx=np.where( (shipdate>=monthend['start'][x]) & (shipdate<=monthend['end'][x] ) )
        shipdate[idx]=monthend['monthend'][x]
    
    return shipdate

