'Purchase order ID'  0
'Vendor ID' 1
'Vendor Name'   2
'Date entered'  3
'Expected date' 4
'Ship VIA'  5
'FOB'   6
'Terms' 7
'Status'    8
'Invoice Number'    9
'Date ordered'  10
'Part SKU'  11
'Part Desc' 12
'Part Category' 13
'Publisher' 14
'Buyer' 15
'Vendor Part#'  16
'Vendor Part description'   17
'Vendor List Price' 18
'Order Quantity'    19
'Received Quantity' 20
'Date of last posting'  21
'Expected delivery date'    22
'Discount% off retail'  23
'Retail price'  24
'Complete flag'  25

import numpy as np
import csv
import numpy.lib.recfunctions as rf
from io import StringIO
np.set_printoptions(suppress=True)

dt = np.dtype([('A0', 'U10'),('A1', 'U10'),('A2', 'U10'),('A3', 'U10'),('A4', 'U10'),('A5', 'U10'),('A6', 'U10'),('A7', 'U10'),('A8', 'U10'),('A9', 'U10'),('A10', 'U10'),('A11', 'U10'),('A12', 'U10'),('A13', 'U10'),('A14', 'U10'),('A15', 'U10'),('A16', 'U10'),('A17', 'U10'),('A18', 'U10'),('A19', 'U10'),('A20', 'U10'),('A21', 'U10'),('A22', 'U10'),('A23', 'U10'),('A24', 'U10'),('A25', 'U10')])

data_list=[]
with open('/home/dfowler/tuples/podata.txt','r') as file:
    reader=csv.reader(file)
    next(reader)
    for row in reader:
        data_list.append(row)

arr=np.array(data_list,dtype=dt)

processed_data = []
for row in reader:
    # Example: Convert numerical fields to float, keep strings as they are
    processed_row = []
    for i, item in enumerate(row):
        try:
            processed_row.append(float(item))
        except ValueError:
            processed_row.append(item) # Keep as string if not a number
            processed_data.append(processed_row)



arr=np.array(processed_data,dtype=dt)

po=np.genfromtxt(data_tuples,delimiter=',',dtype=dt,skip_header=1,maxrows=5)

structured_array = np.array(data_tuples, dtype=my_dtype)





po=np.genfromtxt('/home/dfowler/csv/elasticity.csv',dtype=[('PODATE' ,'U10' ), ('SKU','U13')],usecols=(10,11),delimiter=',',skip_header=1)
