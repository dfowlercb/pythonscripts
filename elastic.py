arr0=np.loadtxt('engine_2017.csv',delimiter=',',usecols=(0,2),quotechar='"',skiprows=1,dtype='i4,U4')
arr0.dtype.names=('oid','engine')
x=np.unique(arr_0['engine'],return_counts=True)
x[0][np.argsort(x[1])[::-1]]
x[1][np.argsort(x[1])[::-1]]
