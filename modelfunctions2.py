def freeze(fname,keycode,maildate,features,idtype,target,start,stop):
    import numpy as np
    import numpy.lib.recfunctions as rf
    
    path=''.join(['/home/dmf/python/npz/rfm',idtype,'.npz'])
    arrays=np.load(path)
    rfm=arrays['rfm']
    if idtype=='hid':
        rfm.dtype.names='cid', 'oidol', 'date', 'church', 'merch', 'hms', 'pro', 'gft', 'fic', 'kds', 'bib', 'aca', 'clv', 'chu', 'mfg', 'mus', 'sea', 'spa', 'vid', 'rec', 'freq', 'mon', 'hmsmon', 'promon', 'gftmon', 'ficmon', 'kdsmon', 'bibmon', 'acamon', 'clvmon', 'chumon', 'mfgmon', 'musmon', 'seamon', 'spamon', 'vidmon', 'score'
    
    fname="".join(['/home/dmf/csv/',fname])
    maildate=np.datetime64(maildate,'D').astype(int)
    mailed=np.genfromtxt(fname,dtype=[('cid',np.int64), ('hid',np.int64),('catalogs','U48') ], delimiter='\t',skip_header=1)
    idx=np.empty(shape=len(mailed['catalogs']),dtype=bool)
    for i in np.arange(len(idx)):
        idx[i]=keycode in mailed['catalogs'][i]
    
    if idtype=='cid':
        idcat=np.unique(mailed['cid'][idx])
    else:
        idcat=np.unique(mailed['hid'][idx])
    
    targetmon=''.join([target,'mon'])
    a=np.logical_and(rfm['date']<=maildate, np.isin(rfm['cid'],idcat) )
    b=np.logical_or(rfm[targetmon]>.01, rfm[target]>.01)
    i=np.logical_and.reduce((a,b))
    rfm=rfm[i]
        
    #target rec, frq stats
    rfmtarget=rfm.copy()
    rfmtarget=rfmtarget[rfmtarget[target]>0]
    i=np.lexsort(( -rfmtarget['oidol'], -rfmtarget['date'], rfmtarget['cid'] ))
    rfmtarget=rfmtarget[i]
    rfmtarget=rf.append_fields(rfmtarget,'frqcount',np.repeat(1,len(rfmtarget['cid'])),usemask=False)
    cid2=np.unique(rfmtarget['cid'])
    frq2=np.bincount(rfmtarget['cid'],weights=rfmtarget['frqcount']).astype(int)
    frq2=frq2[cid2]
    findmax=np.unique(rfmtarget['cid'],return_index=True)
    rec2=maildate-rfmtarget['date'][findmax[1]]
    #
     
    i=np.lexsort(( -rfm['oidol'], -rfm['date'], rfm['cid'] ))
    rfm=rfm[i]
    findmax=np.unique(rfm['cid'],return_index=True)
    rfm=rfm[findmax[1]]
    
    rfm['rec']=maildate-rfm['date']
    rfm['freq']=rfm['freq']+1
    rfm['mon']+=rfm['merch']
    
    for i in ('hms','pro','gft','fic','kds','aca','clv','chu','mfg','mus','sea','spa','vid'):
        x=''.join([i,'mon'])
        rfm[x]+=rfm[i]
    
    #restrictions
    i=np.logical_and(rfm['rec']>=start,rfm['rec']<=stop)
    rfm=rfm[i]
    
    #add fields
    i=np.isin(cid2,rfm['cid'])
    cid2=cid2[i]
    rec2=rec2[i]
    frq2=frq2[i]
    
    i=np.searchsorted(rfm['cid'],cid2)
    cid2=cid2[i]
    rec2=rec2[i]
    frq2=frq2[i]
    tgtpct=np.round(rfm[targetmon]/rfm['mon']*10)
    
    rfm=rf.append_fields(rfm,('rec2','frq2','tgtpct'),(rec2,frq2,tgtpct),dtypes=np.int64,usemask=False) 
    features.extend(('rec2','frq2','tgtpct'))
    
    rfm['rec']=np.round(rfm['rec']/7)
    rfm['rec2']=np.round(rfm['rec2']/7)
     
    return(rfm)

def segmentation(rfmfrz,keycode,idtype,n,features,target):
    import numpy as np
    import numpy.lib.recfunctions as rf
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import MiniBatchKMeans
    kmeans=MiniBatchKMeans(n_clusters=n)
    scaletype = preprocessing.StandardScaler()
     
    X=scaletype.fit_transform(rf.structured_to_unstructured(rfmfrz[features]))
    kmeansclus=kmeans.fit(X)
    predicted_kmeans = kmeans.predict(X)
    rfmfrz=rf.append_fields(rfmfrz,'seg',predicted_kmeans,dtypes=np.int64,usemask=False)
    
    fname=''.join(['/home/dmf/models/',str.lower(keycode),idtype,'_selprior.csv'])
    np.savetxt(fname,rfmfrz,delimiter=',',header=','.join(rfmfrz.dtype.names))
    return(rfmfrz)

def segexport(segment,keycode,target,maildate,idtype):
    import numpy as np
    import numpy.lib.recfunctions as rf
    maildate=np.datetime64(maildate,'D').astype(int)
    cidseg=rf.structured_to_unstructured(segment[['cid','seg']])
    cidseg=cidseg.T
    cidseg.sort(axis=1)
    fname=''.join(['/home/dmf/python/npz/rfm',idtype,'.npz'])
    arrays=np.load(fname)
    rfm=arrays['rfm']
    rfm=rfm[np.where( (rfm['date']>=maildate) & (np.isin(rfm[idtype],cidseg[0])) & (rfm[target]>0) )]
    rfm=rfm[[idtype,target,'merch']]
    
    i=np.searchsorted(cidseg[0],rfm[idtype])
    rfm=rf.append_fields(rfm,'segment',cidseg[1,i],usemask=False)
    bintarget=np.bincount(rfm['segment'],rfm[target])
    binmerch=np.bincount(rfm['segment'],rfm['merch']-rfm[target])
    numcust=np.unique(segment['seg'],return_counts=1)
    
    segf=np.unique(segment['seg']).reshape(-1,1)
    custf=numcust[1].reshape(-1,1)
    salesf1=bintarget.reshape(-1,1)
    salesf2=binmerch.reshape(-1,1)
    salesf3=salesf1/custf
    salesf4=salesf2/custf
    salesf5=salesf3+salesf4
    rank=np.argsort(-salesf3,axis=0)
    rank=rank.flatten()
    rank2=np.arange(len(segf))
    rank2=rank2.reshape(-1,1)
    header=",".join(['rnk','seg','cust','target','nontarget','target/cust','nontarget/cust','total/cust'])
    output=np.hstack((rank2,segf[rank],custf[rank],salesf1[rank],salesf2[rank],salesf3[rank],salesf4[rank],salesf5[rank]))
    fname=''.join(['/home/dmf/models/',str.lower(keycode),idtype,'_segsumprior.csv'])
    np.savetxt(fname,output,delimiter=',',header=header)
    fname=''.join(['/home/dmf/models/',str.lower(keycode),idtype,'_ordered.csv'])
    np.savetxt(fname,rfm,delimiter=',')
    return(output[:,:2])

def knnmodel(segment,features,n,keycode,idtype):
    import numpy as np
    import numpy.lib.recfunctions as rf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn import preprocessing
    from sklearn.neighbors import KNeighborsClassifier
    import numpy.lib.recfunctions as rf
    import joblib
    scaler = preprocessing.StandardScaler()
    knn=KNeighborsClassifier(n_neighbors=n)
    
    X_train, X_test, y_train, y_test = train_test_split(rf.structured_to_unstructured(segment[features]),segment['seg'],random_state=0)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    knn.fit(X_train, y_train)
    print(knn.score(X_test, y_test))
    fname=''.join(['/home/dmf/models/',str.lower(keycode),idtype,'_mod','.txt'])
    joblib.dump(knn,fname)

def rfmtoday(idtype,target,start,stop):
    import numpy as np
    import numpy.lib.recfunctions as rf
    fname=''.join(['/home/dmf/python/npz/rfm',idtype,'.npz'])
    arrays=np.load(fname)
    rfm=arrays['rfm']
    if idtype=='hid':
        rfm.dtype.names='cid', 'oidol', 'date', 'church', 'merch', 'hms', 'pro', 'gft', 'fic', 'kds', 'bib', 'aca', 'clv', 'chu', 'mfg', 'mus', 'sea', 'spa', 'vid', 'rec', 'freq', 'mon', 'hmsmon', 'promon', 'gftmon', 'ficmon', 'kdsmon', 'bibmon', 'acamon', 'clvmon', 'chumon', 'mfgmon', 'musmon', 'seamon', 'spamon', 'vidmon', 'score'
    
    #target rec, frq stats
    rfmtarget=rfm.copy()
    rfmtarget=rfmtarget[rfmtarget[target]>0]
    i=np.lexsort(( -rfmtarget['oidol'], -rfmtarget['date'], rfmtarget['cid'] ))
    rfmtarget=rfmtarget[i]
    rfmtarget=rf.append_fields(rfmtarget,'frqcount',np.repeat(1,len(rfmtarget['cid'])),usemask=False)
    cid2=np.unique(rfmtarget['cid'])
    frq2=np.bincount(rfmtarget['cid'],weights=rfmtarget['frqcount']).astype(int)
    frq2=frq2[cid2]
    findmax=np.unique(rfmtarget['cid'],return_index=True)
    rec2=np.max(rfmtarget['date'])-rfmtarget['date'][findmax[1]]
    #
     
    i=np.lexsort(( -rfm['oidol'], -rfm['date'], rfm['cid'] ))
    rfm=rfm[i]
    findmax=np.unique(rfm['cid'],return_index=True)
    rfm=rfm[findmax[1]]
     
    rfm['rec']=np.max(rfm['date'])-rfm['date']
    rfm['freq']=rfm['freq']+1
    rfm['mon']+=rfm['merch']
    for i in ('hms','pro','gft','fic','kds','aca','clv','chu','mfg','mus','sea','spa','vid'):
        x=''.join([i,'mon'])
        rfm[x]+=rfm[i]
    
    #restrictions
    targetmon=''.join([target,'mon'])
    i0=np.logical_and(rfm['rec']>=start,rfm['rec']<=stop)
    i1=rfm[targetmon]>=1
    i=np.logical_and.reduce((i0,i1))
    rfm=rfm[i]
    
    #add fields
    i=np.isin(cid2,rfm['cid'])
    cid2=cid2[i]
    rec2=rec2[i]
    frq2=frq2[i]
    
    i=np.searchsorted(rfm['cid'],cid2)
    cid2=cid2[i]
    rec2=rec2[i]
    frq2=frq2[i]
    tgtpct=np.round(rfm[targetmon]/rfm['mon']*10)
    
    rfm=rf.append_fields(rfm,('rec2','frq2','tgtpct'),(rec2,frq2,tgtpct),dtypes=np.int64,usemask=False) 
    rfm['rec']=np.round(rfm['rec']/7)
    rfm['rec2']=np.round(rfm['rec2']/7)
     
    return(rfm)

def segfind(keycode,features,rfmnow,daysago,target,idtype):
    import numpy as np
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler
    import joblib
    import numpy.lib.recfunctions as rf
    scaler = preprocessing.StandardScaler()
     
    X=scaler.fit_transform(rf.structured_to_unstructured(rfmnow[features]))
    fname=''.join(['/home/dmf/models/',str.lower(keycode),idtype,'_mod','.txt'])
    knn=joblib.load(fname) 
    y=knn.predict(X)
    rfmnow=rf.append_fields(rfmnow,'seg',y,usemask=False)
    return(rfmnow)

def statsexport(segment,select,n,features,keycode,idtype,rank):
    import numpy as np
    import numpy.lib.recfunctions as rf
    from scipy import stats
    output2=np.zeros(shape=(1,15))
    for i in np.unique(select['seg']):
        prior=stats.describe(rf.structured_to_unstructured(segment[features][segment['seg']==i]))
        future=stats.describe(rf.structured_to_unstructured(select[features][select['seg']==i]))
        modelnum=np.repeat(i,len(prior[1][1]))
        nobsprior=np.repeat(prior[0], len(prior[1][1]))
        nobsfuture=np.repeat(future[0], len(future[1][1]))
        output=np.vstack((modelnum,nobsprior,nobsfuture, prior[1],future[1], prior[2], future[2], prior[3], future[3], prior[4], future[4], prior[5], future[5]))
        output=output.T
        output2=np.vstack((output2,output))
    
    output2=output2[1:,:]
    featurenum=np.tile( np.arange(len(features)) , int(output2.shape[0]/len(features)) ).reshape(-1,1)
    output2=np.hstack( (featurenum , output2 )  )
    rank=rank.astype(int)
    rank=rank[rank[:,1].argsort()]
    i=np.searchsorted(rank[:,1],output2[:,1])
    output2=np.hstack((rank[i,0].reshape(-1,1,),output2))
    i=np.lexsort((output2[:,0],output2[:,1]))
    output2=output2[i,:]
    fname=''.join(['/home/dmf/models/',str.lower(keycode),idtype,'_segsumnext.csv'])
    footer=''.join(np.hstack(( list( zip ( np.arange(len(features)) , np.repeat('-',len(features)),features, np.repeat(',',len(features)) ) ) )))
    np.savetxt(fname,output2,delimiter=',',header='rank,stats,seg,nobsprior,nobsfuture,minprior,maxprior,minnext, maxnext,meanprior,meannext,varianceprior,variancenext,skewnessprior,skewnessnext,kurtosisprior,kurtosisnext', footer=footer)

def selectexport(select,target,keycode,idtype,rank,features):
    import numpy as np
    import numpy.lib.recfunctions as rf
    targetmon=''.join([target,'mon']) 
    rank=rank.astype(int)
    rank=rank[rank[:,1].argsort()]
    i=np.searchsorted(rank[:,1],select['seg'])
    select=rf.append_fields(select,'rank',rank[i,0],usemask=False)
    idx=np.lexsort((-select['mon'],-select['freq'],-select[targetmon],select['rec'],select['rank']))
    select=select[idx]
    features.extend(('rank','seg'))
    select=select[features]
    fname=''.join(['/home/dmf/models/',str.lower(keycode),idtype,'_selnext.txt'])
    np.savetxt(fname,select,delimiter=",",fmt='%f',header=','.join(features))
