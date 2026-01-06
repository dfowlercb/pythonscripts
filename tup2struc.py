#mid replace ,,/,'',/
#bibmid replace ,'', witj ,'0',
#ORDER_NO,ORDER_TYPE,FORM_PAY,BILLTO_ZIPCODE,BILLTO_STATE,NEW_CUST_FLAG,ENTRY_METHD,GENDER,MEMBER_CODE,SKU,KIT_SKU,UNIT_COST,PAH,TAX,LINE_STATUS,LINE_SHIPDATE,CM_ID,ORDER_DATE,HH_ID,CHUR_TAX_CID,CHUR_TAX_HID,UNIT_PRICE,UNITS

#python must be started in ~/python
fname='mtd'
#choose filetype:'hi', 'lohi', 'bib', 'himid','lohimid','orders', 'special'
filetype='lohi'

import numpy as np
import subprocess
import ast
import os
from numpy.lib import recfunctions as rfn

from pyfunc.loads import load_scdhigh as scdh
from pyfunc.loads import load_scdlowhigh as scdlh
from pyfunc.loads import load_bib as bib
from pyfunc.loads import load_orders as ordr
from pyfunc.loads import load_special as spcl

if filetype=='bib':
    subprocess.run(["sed","-i","s/-0/1/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/-1/1/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/-2/1/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/-3/1/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/-4/1/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/-5/1/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/-6/1/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/-7/1/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/-8/1/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/-9/1/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/nf/1/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/Y/1/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/cancelled/0/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/backordered/1/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/shipped/2/g","/home/dfowler/tuples/bib.py"])
    subprocess.run(["sed","-i","s/writeoff/3/g","/home/dfowler/tuples/bib.py"])

tempfiles = {'a':'xaa.py', 'b':'xab.py', 'c':'xac.py', 'd':'xad.py', 'e':'xae.py', 'f':'xaf.py'}

def splitfile(fname):
    os.chdir('/home/dfowler/tuples')
    subprocess.run(['dos2unix','-f',fname])
    subprocess.run(['split','-n','l/6','--additional-suffix=.py', fname])
    with open(fname, 'r') as f:
        newline=f.readline()
        newline=f.readline()
        newline='[\n'+newline
     
    def addheader(fname):
        with open(fname, 'r') as original: data = original.read()
        with open(fname, 'w') as modified: modified.write(newline + data)
     
    header = {key: tempfiles[key] for key in ['b','c','d','e','f']}
    [addheader(v) for v in header.values()]
    
    def addfooter(fname):
        with open(fname, 'a') as file: file.write(']\n')
    
    footer = {key: tempfiles[key] for key in ['a','b','c','d','e']}
    [addfooter(v) for v in footer.values()]

def import_tuple(fname):
    with open(fname,'r') as f:
        data= f.read()
        list0 = ast.literal_eval(data)
        return(list0)

def savescd(lst,arr_name):
    if filetype in ['hi','himid']:
        x=scdh(lst)
    elif filetype in ['lohi','lohimid']:
        x=scdlh(lst)
    elif filetype in ['bib','bibmid']:
        x=bib(lst)
    elif filetype=='orders':
        x=ordr(lst)
    else:
        x=spcl(lst)
     
    x=rfn.drop_fields(x,'KIT SKU',usemask=False)
    fname='/home/dfowler/npz/' + arr_name + '.npz'
    print('Saving array.')
    np.savez_compressed(fname,x)

def stackarr(arr,arr_name):
    fname='/home/dfowler/npz/' + arr_name + '.npz'
    x=np.load(fname)['arr_0']
    arr=rfn.stack_arrays((arr,x),usemask=False)
    return(arr)

def cleanup(fname):
    [os.remove('/home/dfowler/tuples/'+v) for v in tempfiles.values()]
    [os.remove('/home/dfowler/npz/'+fname+k+'.npz')  for k in tempfiles.keys()]

def processfile(fname):
    fname1=fname + '.py'
    splitfile(fname1)
    for k,v in tempfiles.items(): 
        print('Processing chunk:'+v)
        list0=import_tuple(v)
        savescd(list0,fname+k)
    print('Stacking arrays')
    stack = {key: tempfiles[key] for key in ['b','c','d','e','f']}
    arr_0=np.load('/home/dfowler/npz/'+fname+'a.npz')['arr_0']
    [arr_0:=stackarr(arr_0,fname+k) for k in stack.keys()] 
    arr_0=np.sort(arr_0,order=('ORDER_NO','SKU'))
    print('Saving file:'+fname+'.npz')
    np.savez_compressed('/home/dfowler/npz/'+fname+'.npz',arr_0)
    cleanup(fname)

processfile(fname)

