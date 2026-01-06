import numpy as np
import pandas as pd
# remove =," from cb.csv via vim first

file_instance = pd.ExcelFile('/home/dmf/csv/nyt.xlsx')
arr0 = pd.concat([pd.read_excel('/home/dmf/csv/nyt.xlsx', sheet_name=name) for name in file_instance.sheet_names] , axis=0)
arr0=arr0['ISBN'].to_numpy()
arr1 = np.genfromtxt('/home/dmf/csv/cb.csv',delimiter=',',skip_header=1,dtype=np.int64,usecols=(0,1))
arr1=arr1[np.isin(arr1[:,0],arr0)]
arr1=arr1[arr1[:,1].argsort()[::-1]]
np.savetxt('/home/dmf/Documents/x.csv',arr1,delimiter=',',header='isbn,quantity',comments='',fmt='%i')

