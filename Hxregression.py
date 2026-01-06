import numpy as np
import numpy.lib.recfunctions as rf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
np.set_printoptions(suppress=True)

Hx=np.genfromtxt('/home/dfowler/csv/elasticity.csv',dtype=[('SKU' ,'U13' ), ('T1','U13'), ('CB' ,'f8' ), ('EV' ,'f8' ),('AZ' ,'f8' ), ('WM' ,'f8' )],usecols=(1,3,11,29,31,32),delimiter=',',skip_header=1,comments='}')
parts=np.genfromtxt('/home/dfowler/csv/parts.txt',dtype=[('part' ,'U13' ), ('BG' ,'U20' ), ('RETAIL' ,'f8' )],delimiter=',',encoding='latin1',usecols=(6,12,31),skip_header=1,filling_values=0.)
bisac=np.genfromtxt('/home/dfowler/csv/bisac.csv',dtype=[('SKU','U13'),('BISAC1','U24'),('BISAC2','U24'),('BISAC3','U24')],usecols=(0,1,2,3),delimiter=',',skip_header=1)

Hx['T1'][np.char.startswith(Hx['T1'],'09')]='BOOKS'
Hx['T1'][np.isin(Hx['T1'],['BOOKS'],invert=True)]='NON-BOOKS'

parts=np.sort(parts,order='part')
idx=np.searchsorted(parts['part'],Hx['SKU'])
BG=parts['BG'][idx].reshape(-1,1)
RETAIL=parts['RETAIL'][idx].reshape(-1,1)
T1=Hx['T1'].reshape(-1,1)
Hx=rf.append_fields(Hx,('BG','RETAIL'),(BG,RETAIL),usemask=False)

bisac=np.sort(bisac,order='SKU')
idx=np.searchsorted(bisac['SKU'],Hx['SKU'])
BISAC1=bisac['BISAC1'][idx].reshape(-1,1)
BISAC2=bisac['BISAC2'][idx].reshape(-1,1)
BISAC3=bisac['BISAC3'][idx].reshape(-1,1)
ohe = OneHotEncoder(sparse_output=False)
BG_encoded=ohe.fit_transform(BG)
T1_encoded=ohe.fit_transform(T1)
BISAC1_encoded=ohe.fit_transform(BISAC1)
BISAC2_encoded=ohe.fit_transform(BISAC2)
BISAC3_encoded=ohe.fit_transform(BISAC3)

#AZ & WM
#x=rf.structured_to_unstructured(Hx[['RETAIL','CB','AZ','WM','EV']])
#features=np.array(['RETAIL','CB','WM','EV'])
#features=np.hstack((features,np.unique(BG)))

#NO WM PRICE
arr1=rf.structured_to_unstructured(Hx[['RETAIL','CB','AZ','EV']])
features=np.array(['RETAIL','CB','EV'])
features=np.hstack((features,np.unique(BG),np.unique(T1),np.unique(BISAC1),np.unique(BISAC2),np.unique(BISAC3)))

arr1=np.hstack((arr1,BG_encoded,T1_encoded,BISAC1_encoded,BISAC2_encoded,BISAC3_encoded))
idx=~np.isnan(arr1).any(axis=1)
data=arr1[idx]
Hx=Hx[idx]
indices=Hx['SKU'].reshape(-1,1)
#imp = IterativeImputer(max_iter=10, random_state=0)
#imp.fit(x)
#x=imp.transform(x)
target=data[:,2].reshape(-1,1)
#imp.fit(y)
#y=imp.transform(y)
data=np.delete(data,2,1)
x_train, x_test, y_train, y_test,indices_train,indices_test = train_test_split(data,target,indices)

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x_train, y_train)
print('Linear Regression Train',model.score(x_train, y_train))
print('Linear Regression Test',model.score(x_test, y_test))

from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor().fit(x_train, np.ravel(y_train))
print('Gradient Boosting Regressor Train',model.score(x_train, y_train))
print('Gradient Boosting Regressor Test',model.score(x_test, y_test))

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor().fit(x_train, np.ravel(y_train))
print('Random Forest Regressor Train ',model.score(x_train, y_train))
print('Random Forest Regressor Test ',model.score(x_test, y_test))

from sklearn.tree import DecisionTreeRegressor
# Initialize the regressor
regressor = DecisionTreeRegressor(max_depth=7)
# Train the regressor on the training data
decision_tree=regressor.fit(x_train, y_train)
print('Decision Tree Regressor Train ',decision_tree.score(x_train, y_train))
print('Decision Tree Regressor Test ',decision_tree.score(x_test, y_test))
# Make predictions on the test data
y_pred = regressor.predict(x_test)

#from sklearn.tree import plot_tree
#import matplotlib.pyplot as plt

# Plot the decision tree
#plt.figure(figsize=(75,75))
#plot_tree(regressor, filled=True, max_depth=10, fontsize=6, feature_names=features)
#plt.savefig('/home/dfowler/documents/tree.png',dpi=100)

#from sklearn.tree import export_text
#r = export_text(decision_tree, feature_names=features,show_weights=True)
#with open(r'/home/dfowler/documents/tree.txt', 'w') as file_object:
#        print(r, file=file_object)

import dtreeviz

titlename=indices_train[3]

viz=dtreeviz.model(decision_tree,x_train,y_train,feature_names=features,target_name='AZ')
v = viz.view(fontname='monospace',orientation='TD',fancy=False)  
v.save('/home/dfowler/documents/treeplain.svg')

viz=dtreeviz.model(decision_tree,x_train,y_train,feature_names=features,target_name='AZ')
v = viz.view(fontname='monospace',orientation='TD',show_node_labels=True,fancy=True)  
v.save('/home/dfowler/documents/treefancy.svg')

viz=dtreeviz.model(decision_tree,x_train,y_train,feature_names=features,target_name='AZ')
v = viz.view(x=x_train[3],fontname='monospace',fancy=False,show_just_path=False,title=titlename)  
v.save('/home/dfowler/documents/treepath1.svg')
print(indices_train[3])

viz=dtreeviz.model(decision_tree,x_train,y_train,feature_names=features,target_name='AZ')
v = viz.view(x=x_train[3],fontname='monospace',fancy=False,show_just_path=True,title=titlename)  
v.save('/home/dfowler/documents/treepath2.svg')
#viz_model.view(depth_range_to_display=(1, 2)) # root is level 0

import os
os.chdir('/home/dfowler/documents')
#os.system('rclone copy tree.png onedrive:machine_learning')
#os.system('rclone copy tree.txt onedrive:machine_learning')
os.system('rclone copy treeplain.svg onedrive:machine_learning')
os.system('rclone copy treefancy.svg onedrive:machine_learning')
os.system('rclone copy treepath1.svg onedrive:machine_learning')
os.system('rclone copy treepath2.svg onedrive:machine_learning')
