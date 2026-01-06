def load(levelmonth,*features):
    """Loads specified arrays from the monthly file.
    
    Parameters: 
    levelmonth: 
    ol + 3 letter month for order level; 
    ll + 3 letter month for line
    
    Returns:
    Specified arrays. 
    """
    import os
    import numpy as np
    os.chdir('/home/dmf/python/npz')
    fname=''.join([levelmonth,'hilo.npz'])
    arrays=np.load(fname,allow_pickle=True)
    for x in np.arange(len(features)):
        globals()[features[x]]=arrays[features[x]]

def scope(idx,*args):
    """Reduces the scope of an array(s) based on an index.

    Parameters:
    idx: Statement that returns indices of desired scope.
    features: One or more features (arrays) that will be reduced to the index size.
    
    Returns: The features are reduced to the index size.
    """
    for i in np.arange(len(args)):
        globals()[args[i]]=globals()[args[i]][idx]


