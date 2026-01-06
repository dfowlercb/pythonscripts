import numpy as np
from modelfunctions2 import freeze, segmentation, segexport, knnmodel, rfmtoday, segfind, statsexport, selectexport

fname='mailed.txt'
keycode='DWS'
maildate='2022-01-07'
start,stop=0,365
numberofsegments=48
idtype='cid'
features=['rec','freq','mon','hms','promon','hmsmon','chumon','church']
target='hms'

#run model functions
rfmfrz=freeze(fname,keycode,maildate,features,idtype,target,start,stop)
segment=segmentation(rfmfrz,keycode,idtype,numberofsegments,features,target)
rank=segexport(segment,keycode,target,maildate,idtype)
knnmodel(segment,features,5,keycode,idtype)

rfmnow=rfmtoday(idtype,target,start,stop)
select=segfind(keycode,features,rfmnow,365*1,target,idtype)
statsexport(segment,select,numberofsegments,features,keycode,idtype,rank)
selectexport(select,target,keycode,idtype,rank,features)

#USER INPUT
#fname=input('Enter file name containing list of customers that were mailed:  ')
#keycode=input('Enter 3 letter keycode of catalog for auto-segment:  ')
#maildate=input('Enter maildate of catalog for auto-segment in yyyy-mm-dd format:  ')
#numberofsegments=int(input('Enter number of desired segments:  '))
#idtype=input('Enter cid for customer basis or hid for household basis:  ')

#features = []
#while True:
    #inp = input('Enter features one at a time and use empty return when finished:  ')
    #if inp == "":
        #break
    #features.append(inp)

#target=input('Enter 3 letter category code for targeted sales dollars:  ')

