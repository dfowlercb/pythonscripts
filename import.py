
remove embedded nuls from text file %s/ctrl+@//g
#INSTALL PIP
sudo apt update
sudo apt install python3-zip
pip 3 --version

#UPDATE PIP
python3 -m pip install --user --upgrade pip
python3 -m pip --version

#INSTALL VENV
sudo apt install python3.9-venv

#INSTALL  
pip install pandas
pip install numpy
pip install xlrd #for reading excel

#VIRTUAL ENVIRONMENT
#CD TO DIRECTORY FOR PROJECT
python3 -m venv env
source env/bin/activate
which python

#leave environment
deactivate

#libraries
import pandas as pd
import numpy as np

#IMPORT TEXT
orders=pd.read_csv('orders.txt',sep='\t',encoding='latin-1')
orders.head()

#LIST COLUMN NAMES
list(orders.columns)

#LIST COLUMN TYPES
orders.dtypes

#LIST OBJECTS LIKE R ls()
dir()

#SELECT COLUMNS
orders[['BILLTO STATE','UNIT_PRICE','UNITS']]

#RENAME COLUMNS
orders=orders.rename(columns={'BILLTO STATE':'STATE'})

#CREATE COLUMNS
orders['SALES']=(orders['UNITS']*orders['UNIT_PRICE'])

#DELETE COLUMNS
orders=orders.drop(columns=['FORM_PAY'])

#FILTER
orders.loc[orders['STATE']=='NH', ['STATE','SALES']]

#ARITHMETIC
orders['UNITS'].sum()
orders['UNITS'].mean()
orders['UNITS'].max()
orders['UNITS'].min()
orders['UNITS'].std()

#COUNT, UNIQUE COUNT
nh_orders=orders[orders['BILLTO STATE']=='NH']
nh_orders['CM_ID'].count()
nh_orders['CM_ID'].nunique()

#GROUP BY
orders.groupby('BILLTO STATE')['CM_ID'].nunique()
orders[orders['BILLTO STATE']=='NH'].groupby(['BILLTO STATE','ORDER_TYPE'])['CM_ID'].nunique()
orders[orders.STATE.isin(['NH','MA'])].groupby(['STATE','ORDER_TYPE'])['CM_ID'].nunique()
orders.groupby('STATE').agg({'UNITS':['sum', 'max'],
                         'UNIT_PRICE':'mean',
                         'CM_ID':'nunique'})

#PIVOT TABLE

orders.pivot_table(index=['STATE','ORDER_TYPE'],columns='FORM_PAY',aggfunc=np.sum,values='UNITS')

x=orders.pivot_table(values=['UNITS','UNIT_PRICE'],index=['STATE'],columns='FORM_PAY',aggfunc={'UNITS':np.sum,'UNIT_PRICE':[min,max,np.mean]})

#EXPORT TO CSV
x.to_csv('~/python/practice/x.txt',sep='|',index=True)

#DATAFRAME TO ARRAY
y=x.to_numpy

#JOINS
parts=pd.read_csv('parts.txt',sep=',',encoding='latin-1',usecols=['part','category','publisher'])
orders=orders.join(parts.set_index('part'),on='SKU')

#COMBINE DATA FRAMES
dt0=pd.concat([dt16,dt17,dt18])

#STRING STARTS WITH
df.loc[(df['publisher'].str.startswith('LRS',na=False)),['SKU','publisher']]


#DATE RANGE BIN / MASK
mask = (df['DATE'] >='2020-07-01' ) & (df['DATE'] <= '2021-06-30')
df.loc[mask,'TTM']='JUL20-JUN21'

#CORRELATION MATRIX
x=orders.pivot_table(values='SALES',index='CM_ID',columns
y=corr(x)


#TTM REPORT
mask = (df['DATE'] >='2020-07-01' ) & (df['DATE'] <= '2021-06-30')
df.loc[mask,'TTM']='JUL20-JUN21'
mask = (df['DATE'] >='2019-07-01' ) & (df['DATE'] <= '2020-06-30')
df.loc[mask,'TTM']='JUL19-JUN20'
mask = (df['DATE'] >='2018-07-01' ) & (df['DATE'] <= '2019-06-30')
df.loc[mask,'TTM']='JUL18-JUN19'
mask = (df['DATE'] >='2017-07-01' ) & (df['DATE'] <= '2018-06-30')
df.loc[mask,'TTM']='JUL17-JUN18'
mask = (df['DATE'] >='2016-07-01' ) & (df['DATE'] <= '2017-06-30')
df.loc[mask,'TTM']='JUL16-JUN17'

