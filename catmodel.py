import numpy as np
import numpy.lib.recfunctions as rf
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
import joblib

class CatModel:
    def __init__(self,fname,keycode,maildate,start,stop,numberofsegments,idtype,features,target):
        self.fname=fname
        self.keycode=keycode
        self.maildate=maildate
        self.start=start
        self.stop=stop
        self.numberofsegments=numberofsegments
        self.idtype=idtype
        self.features=features
        self.target=target

    def _freeze(self,fname,keycode,maildate,features,idtype,target,start,stop):
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
    
    def segmentation(self):
        rfmfrz=self._freeze(self.fname,self.keycode,self.maildate,self.features,self.idtype,self.target,self.start,self.stop)
        kmeans=MiniBatchKMeans(n_clusters=self.numberofsegments)
        scaletype = preprocessing.StandardScaler()
        
        X=scaletype.fit_transform(rf.structured_to_unstructured(rfmfrz[self.features]))
        kmeansclus=kmeans.fit(X)
        predicted_kmeans = kmeans.predict(X)
        rfmfrz=rf.append_fields(rfmfrz,'seg',predicted_kmeans,dtypes=np.int64,usemask=False)
        
        fname=''.join(['/home/dmf/models/',str.lower(self.keycode),self.idtype,'_selprior.csv'])
        np.savetxt(fname,rfmfrz,delimiter=',',header=','.join(rfmfrz.dtype.names),fmt='%1.3f',comments='')
        
        def _segexport():
            maildate=np.datetime64(self.maildate,'D').astype(int)
            cidseg=rf.structured_to_unstructured(rfmfrz[['cid','seg']])
            cidseg=cidseg.T
            cidseg.sort(axis=1)
            fname=''.join(['/home/dmf/python/npz/rfm',self.idtype,'.npz'])
            arrays=np.load(fname)
            rfm=arrays['rfm']
            rfm=rfm[np.where( (rfm['date']>=maildate) & (np.isin(rfm[self.idtype],cidseg[0])) & (rfm[self.target]>0) )]
            rfm=rfm[[self.idtype,'oidol',self.target,'merch']]
            
            i=np.searchsorted(cidseg[0],rfm[self.idtype])
            rfm=rf.append_fields(rfm,'segment',cidseg[1,i],usemask=False)
            bintarget=np.bincount(rfm['segment'],rfm[self.target])
            binmerch=np.bincount(rfm['segment'],rfm['merch']-rfm[self.target])
            numcust=np.unique(rfmfrz['seg'],return_counts=1)
            
            segf=np.unique(rfmfrz['seg']).reshape(-1,1)
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
            output=np.hstack((rank2,segf[rank],custf[rank],salesf1[rank],salesf2[rank],salesf3[rank],salesf4[rank],salesf5[rank]))
            
            fname=''.join(['/home/dmf/models/',str.lower(self.keycode),self.idtype,'_segsumprior.csv'])
            header=",".join(['rnk','seg','cust','target','nontarget','target/cust','nontarget/cust','total/cust'])
            np.savetxt(fname,output,delimiter=',',header=header,fmt='%1.3f',comments='')
            
            fname=''.join(['/home/dmf/models/',str.lower(self.keycode),self.idtype,'_ordered.csv'])
            header=",".join(['cid','oid','target','total'])
            np.savetxt(fname,rfm,delimiter=',',fmt='%1.3f',comments='',header=header)
            
            return rfmfrz,output[:,:2]
        
        self.segment, self.rank=_segexport()
            
        def _knnmodel():
            scaler = preprocessing.StandardScaler()
            knn=KNeighborsClassifier(n_neighbors=5)
            
            X_train, X_test, y_train, y_test = train_test_split(rf.structured_to_unstructured(rfmfrz[self.features]),rfmfrz['seg'],random_state=0)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            knn.fit(X_train, y_train)
            print(knn.score(X_test, y_test))
            fname=''.join(['/home/dmf/models/',str.lower(self.keycode),self.idtype,'_mod','.txt'])
            joblib.dump(knn,fname)
    
        _knnmodel()
    
    def _rfmnow(self):
        fname=''.join(['/home/dmf/python/npz/rfm',self.idtype,'.npz'])
        arrays=np.load(fname)
        rfm=arrays['rfm']
        if self.idtype=='hid':
            rfm.dtype.names='cid', 'oidol', 'date', 'church', 'merch', 'hms', 'pro', 'gft', 'fic', 'kds', 'bib', 'aca', 'clv', 'chu', 'mfg', 'mus', 'sea', 'spa', 'vid', 'rec', 'freq', 'mon', 'hmsmon', 'promon', 'gftmon', 'ficmon', 'kdsmon', 'bibmon', 'acamon', 'clvmon', 'chumon', 'mfgmon', 'musmon', 'seamon', 'spamon', 'vidmon', 'score'
    
        #target rec, frq stats
        rfmtarget=rfm.copy()
        rfmtarget=rfmtarget[rfmtarget[self.target]>0]
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
        targetmon=''.join([self.target,'mon'])
        i0=np.logical_and(rfm['rec']>=self.start,rfm['rec']<=self.stop)
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
 
    def segfind(self):
        rfmnow=self._rfmnow()
        scaler = preprocessing.StandardScaler()
        X=scaler.fit_transform(rf.structured_to_unstructured(rfmnow[self.features]))
        fname=''.join(['/home/dmf/models/',str.lower(self.keycode),self.idtype,'_mod','.txt'])
        knn=joblib.load(fname) 
        y=knn.predict(X)
        rfmnow=rf.append_fields(rfmnow,'seg',y,usemask=False)
        
        def _statsexport():
            output2=np.zeros(shape=(1,15))
            for i in np.unique(rfmnow['seg']):
                prior=stats.describe(rf.structured_to_unstructured(self.segment[self.features][self.segment['seg']==i]))
                future=stats.describe(rf.structured_to_unstructured(rfmnow[self.features][rfmnow['seg']==i]))
                modelnum=np.repeat(i,len(prior[1][1]))
                nobsprior=np.repeat(prior[0], len(prior[1][1]))
                nobsfuture=np.repeat(future[0], len(future[1][1]))
                output=np.vstack((modelnum,nobsprior,nobsfuture, prior[1],future[1], prior[2], future[2], prior[3], future[3], prior[4], future[4], prior[5], future[5]))
                output=output.T
                output2=np.vstack((output2,output))
            
            output2=output2[1:,:]
            featurenum=np.tile( np.arange(len(self.features)) , int(output2.shape[0]/len(self.features)) ).reshape(-1,1)
            output2=np.hstack( (featurenum , output2 )  )
            rank=self.rank.astype(int)
            rank=rank[rank[:,1].argsort()]
            i=np.searchsorted(rank[:,1],output2[:,1])
            output2=np.hstack((rank[i,0].reshape(-1,1,),output2))
            i=np.lexsort((output2[:,0],output2[:,1]))
            output2=output2[i,:]
            fname=''.join(['/home/dmf/models/',str.lower(self.keycode),self.idtype,'_segsumnext.csv'])
            footer=''.join(np.hstack(( list( zip ( np.arange(len(self.features)) , np.repeat('-',len(self.features)),self.features, np.repeat(',',len(self.features)) ) ) )))
            np.savetxt(fname,output2,delimiter=',',header='rank,stats,seg,nobsprior,nobsfuture,minprior,maxprior,minnext,maxnext,meanprior,meannext,varianceprior,variancenext,skewnessprior,skewnessnext,kurtosisprior,kurtosisnext', footer=footer,fmt='%1.3f',comments='')
            return(rfmnow) 
        
        self.rfmnow=_statsexport()
         
        def _selectexport():
            targetmon=''.join([self.target,'mon']) 
            rank=self.rank.astype(int)
            rank=rank[rank[:,1].argsort()]
            i=np.searchsorted(rank[:,1],self.rfmnow['seg'])
            self.rfmnow=rf.append_fields(self.rfmnow,'rank',rank[i,0],usemask=False)
            idx=np.lexsort((-self.rfmnow['mon'],-self.rfmnow['freq'],-self.rfmnow[targetmon],self.rfmnow['rec'],self.rfmnow['rank']))
            self.rfmnow=self.rfmnow[idx]
            self.features.extend(('rank','seg'))
            self.rfmnow=self.rfmnow[self.features]
            fname=''.join(['/home/dmf/models/',str.lower(self.keycode),self.idtype,'_selnext.txt'])
            np.savetxt(fname,self.rfmnow,delimiter=",",fmt='%1.3f',comments='',header=','.join(self.features))
         
        _selectexport()


