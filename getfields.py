import numpy as np

def get_fields(name,delim):
    loc=str('/home/dmf/csv/')
    fname=loc+name
    fields=np.genfromtxt(fname,delimiter=delim,max_rows=1,dtype='U50')
    fields=np.hstack([np.arange(len(fields)).reshape(-1,1)  , fields.reshape(-1,1) ])
    return(fields) 


