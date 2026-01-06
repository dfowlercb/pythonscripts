import numpy as np
import ast
from pyfunc.loads import load_shipcost as lsc

def import_csv(fname):
        with open(fname,'r') as f:
            data= f.read()
            list0 = ast.literal_eval(data)
            return(list0)

list0=import_csv('/home/dfowler/csv/shipcostapr23.csv')
