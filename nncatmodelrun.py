from nncatmodel import CatModel as cm 

fname='mailed.txt'
keycode='DWS'
maildate='2022-01-07'
start,stop=0,365
numberofsegments=48
idtype='cid'
features=['rec','freq','mon','hms','promon','hmsmon','chumon','church']
target='hms'

model=cm(fname,keycode,maildate,start,stop,numberofsegments,idtype,features,target)
model.segmentation()
model.segfind()
