import pandas as pd
import numpy as np

#IMPORT AND PROCESS ORDER FILES
o12=pd.read_csv('~/R/projects/consensus/skucodedetails2012.txt',sep='\t',encoding='latin-1',dtype={'CM_ID':str,'HOUSEHOLD':str,'BILLTO_ZIPCODE':str})
o13=pd.read_csv('~/R/projects/consensus/skucodedetails2013.txt',sep='\t',encoding='latin-1',dtype={'CM_ID':str,'HOUSEHOLD':str,'BILLTO_ZIPCODE':str})
o14=pd.read_csv('~/R/projects/consensus/skucodedetails2014.txt',sep='\t',encoding='latin-1',dtype={'CM_ID':str,'HOUSEHOLD':str,'BILLTO_ZIPCODE':str})
o15=pd.read_csv('~/R/projects/consensus/skucodedetails2015.txt',sep='\t',encoding='latin-1',dtype={'CM_ID':str,'HOUSEHOLD':str,'BILLTO_ZIPCODE':str})
o16=pd.read_csv('~/R/projects/consensus/skucodedetails2016.txt',sep='\t',encoding='latin-1',dtype={'CM_ID':str,'HOUSEHOLD':str,'BILLTO_ZIPCODE':str})
df=pd.concat([o12,o13,o14,o15,o16])

o16=pd.read_csv('~/R/projects/consensus/skucodedetails2016.txt',sep='\t',encoding='latin-1',dtype={'CM_ID':str,'HOUSEHOLD':str,'BILLTO_ZIPCODE':str})
o17=pd.read_csv('~/R/projects/consensus/skucodedetails2017.txt',sep='\t',encoding='latin-1',dtype={'CM_ID':str,'HOUSEHOLD':str,'BILLTO_ZIPCODE':str})
o18=pd.read_csv('~/R/projects/consensus/skucodedetails2018.txt',sep='\t',encoding='latin-1',dtype={'CM_ID':str,'HOUSEHOLD':str,'BILLTO_ZIPCODE':str})
o19=pd.read_csv('~/R/projects/consensus/skucodedetails2019.txt',sep='\t',encoding='latin-1',dtype={'CM_ID':str,'HOUSEHOLD':str,'BILLTO_ZIPCODE':str})
o20=pd.read_csv('~/R/projects/consensus/skucodedetails2020.txt',sep='\t',encoding='latin-1',dtype={'CM_ID':str,'HOUSEHOLD':str,'BILLTO_ZIPCODE':str})
o21=pd.read_csv('~/R/projects/consensus/skucodedetails2021.txt',sep='\t',encoding='latin-1',dtype={'CM_ID':str,'HOUSEHOLD':str,'BILLTO_ZIPCODE':str})
df=pd.concat([o16,o17,o18,o19,o20,o21])

df=df[~(df['LINE STATUS']=='cancelled')]
df=df[df.ORDER_TYPE.isin(['W','P','M'])]
df['DATE']=pd.to_datetime(df['ORDER_DATE'])
df['YEAR']=pd.DatetimeIndex(df['DATE']).year
df['MONTH']=pd.DatetimeIndex(df['DATE']).month

#ADD MSA
zipcode=pd.read_excel('zipcountymsa.xls',dtype={'ZIP':str})
zipcode=zipcode.drop_duplicates(subset=['ZIP'])
zipcode.CBSANAME2=zipcode.CBSANAME2.fillna('')
zipcode=zipcode[['ZIP','CBSANAME2']]
zipcode.sort_values(by=['ZIP'])

zipcode.set_index('ZIP')
df.set_index('BILLTO_ZIPCODE')
df=df.merge(zipcode, how='left', left_on='BILLTO_ZIPCODE', right_on='ZIP')
df.CBSANAME2=df.CBSANAME2.fillna('')
df['SALES']=df['UNITS']*df['UNIT_PRICE']            

#ADD LIFEWAY
lifeway=pd.read_csv('lifeway.csv',dtype={'lifeway':str})
lifeway=lifeway.rename(columns={'lifeway':'ZIP'})
lifeway=lifeway.merge(zipcode,how='left',left_on='ZIP',right_on='ZIP')
lifeway=lifeway.drop(columns=['ZIP'])
lifeway['CBSANAME2'].replace('',np.nan,inplace=True)
lifeway=lifeway.dropna()
df['LIFEWAY']='N'
stores=lifeway['CBSANAME2'].unique()
df.loc[(df['CBSANAME2'].isin(stores)),'LIFEWAY']='Y'

#CREATE REPORT AND EXPORT
#pd.options.display.float_format = '{:.2f}'.format
#pd.reset_option('display.float_format')
x=df.pivot_table(values='SALES',index='LIFEWAY',columns='YEAR',aggfunc=sum)
x.to_csv('~/python/practice/x.txt',sep='|',index=True)
y=df[df['LIFEWAY']=='Y'].pivot_table(values='SALES',index='CBSANAME2',columns='YEAR',aggfunc=sum)
y.to_csv('~/python/practice/y.txt',sep='|',index=True)

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

x=df.pivot_table(values='SALES',index='LIFEWAY',columns='TTM',aggfunc=sum)
x.to_csv('~/python/practice/x.txt',sep='|',index=True)
y=df[df['LIFEWAY']=='Y'].pivot_table(values='SALES',index='CBSANAME2',columns='TTM',aggfunc=sum)
y.to_csv('~/python/practice/y.txt',sep='|',index=True)
