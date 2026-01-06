#requires skus array to be loaded
import numpy as np
from numpy import char
from numpy import genfromtxt

parts=genfromtxt('/home/dmf/csv/parts.txt',skip_header=1,delimiter=',',dtype='str',encoding='latin1',usecols=(6,11,12),converters={6: lambda s: str(s or ''),11: lambda s: str(s or ''),12: lambda s: str(s or '')})
parts=parts[parts[:,0].argsort()]
tiers=char.partition(parts[:,1],'-')
tier1=tiers[:,0]
tiers=char.partition(tiers[:,2],'-')
tier2=tiers[:,0]
tier3=tiers[:,2]
del tiers
parts=np.vstack((parts.T,tier1,tier2,tier3))
partslookup=np.searchsorted(parts[0],skus)

cat=parts[1][partslookup]
categories=np.unique(parts[1])
catlookup=np.searchsorted(categories,cat)

bg=parts[2][partslookup]
buyergroups=np.unique(parts[2])
bglookup=np.searchsorted(buyergroups,bg)

t1=parts[3][partslookup]
tier1=np.unique(parts[3])
t1lookup=np.searchsorted(tier1,t1)

t2=parts[4][partslookup]
tier2=np.unique(parts[4])
t2lookup=np.searchsorted(tier2,t2)

t3=parts[5][partslookup]
tier3=np.unique(parts[5])
t3lookup=np.searchsorted(tier3,t3)

catdict=dict(enumerate(categories.flatten()))
catdict=np.array(list(catdict.items()))
bgdict=dict(enumerate(buyergroups.flatten()))
bgdict=np.array(list(bgdict.items()))
t1dict=dict(enumerate(tier1.flatten()))
t1dict=np.array(list(t1dict.items()))
t2dict=dict(enumerate(tier2.flatten()))
t2dict=np.array(list(t2dict.items()))
t3dict=dict(enumerate(tier3.flatten()))
t3dict=np.array(list(t3dict.items()))

np.savez_compressed('/home/dmf/python/npz/parts.npz',parts=partslookup,cat=catlookup,bg=bglookup,t1=t1lookup,t2=t2lookup,t3=t3lookup,catdict=catdict,t1dict=t1dict,t2dict=t2dict,t3dict=t3dict,bgdict=bgdict)
del(parts,cat,bg,t1,t2,t3,categories,buyergroups,tier1,tier2,tier3)
