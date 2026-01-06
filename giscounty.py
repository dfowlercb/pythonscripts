import pandas as pd
import numpy as np
import csv
df=pd.read_csv('~/csv/skucodedetails2020.txt',sep='\t',encoding='latin-1',dtype={'CM_ID':str,'HOUSEHOLD':str,'BILLTO_ZIPCODE':str})
zipcty=pd.read_csv('~/gis/geotables/zipcountymsa.csv',sep='\t',dtype={'ZIP':str,'STCTY':str},usecols=['ZIP','STCTY','POP10'])
zipcty=zipcty.drop_duplicates(subset=['ZIP'])
df=df.set_index('BILLTO_ZIPCODE',drop=False)
zipcty=zipcty.set_index('ZIP')
df=df.join(zipcty,how='left')
df['SALES']=df['UNITS']*df['UNIT_PRICE']

parts=pd.read_csv('~/csv/parts.txt',sep=',',encoding='latin-1',usecols=['part','BUYER_GROUP'])
parts=parts.drop_duplicates(subset=['part'])
parts=parts.set_index('part')
df=df.set_index('SKU')
df=pd.merge(df,parts,how='left',left_index=True,right_index=True)

x=df.query('BUYER_GROUP=="CHURCH INTEREST"').pivot_table(values=['SALES'],index='STCTY',aggfunc=sum)
x.to_csv('~/gis/attributes/stctyCHURCHBG.csv',sep='\t',quoting=csv.QUOTE_ALL)
#CREATE stcty.csvt in same directory with two fields: "string","real" for QGIS import with leading zeros on zipcode
csvt=pd.DataFrame(columns=['string','real'])
csvt.to_csv('~/gis/attributes/stctyCHURCHBG.csvt',sep=",",index=False,quoting=csv.QUOTE_ALL)


#QGIS
#CREATE VIRTUAL JOIN TEXT FIELD ON COUNTY LAYER concat(to_string(STATEFP),to_string(COUNTYFP))
#RIGHT CLICK on layer to filter by state, county, etc. OR choose layer add virtual layer and use SQL query to keep whole map but chorpleth an individual state i.e. SELECT * FROM countylayer WHERE STATEUSPS='NH'

