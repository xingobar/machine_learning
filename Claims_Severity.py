 # http://stackoverflow.com/questions/15723628/pandas-make-a-column-dtype-object-or-factor
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import skew
from scipy.sparse import csr_matrix,hstack
from scipy.stats import boxcox
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.grid_search import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
#from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
import datetime
import xgboost as xgb
import time

#def fmean_squared_error(ground_truth,prediction):
    
#    fmean_squared_error_ = mean_sauared_error(ground_truth,prediction)**0.5
#    return fmean_squared_error_


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test_ids = test.id
target = np.log1p(train.loss.values)
num_trains = train.shape[0]
#RMSE = make_score(fmean_squared_error,greater_is_better=False)
df_train = train.drop(['id','loss'],axis=1)
df_test = test.drop(['id'],axis=1)
all_data = pd.concat((df_train,df_test),axis=0).reset_index(drop=True)
features = all_data.columns
cat_list = [feat for feat in features if 'cat' in feat]
cont_list =[feat for feat in features if 'cont' in feat]

numeric_feat = all_data.dtypes[all_data.dtypes != 'object'].index
skewed = all_data[numeric_feat].apply(lambda x:skew(x.dropna()))
skewed_less = skewed[skewed <= 0.25].index
skewed = skewed[skewed > 0.25].index
'''
all_data[skewed] = np.log1p(all_data[skewed])
all_data[skewed_less] = np.exp(all_data[skewed_less])
all_data['cont1'] = all_data[all_data['cont1'] >= 0.3930894470200948]['cont1']
all_data['cont9'] = all_data[all_data['cont9'] >= 0.38823598271681442]['cont9']
all_data['cont10'] = all_data[all_data['cont10'] >= 0.39602509335206326]['cont10']
'''


boxcox_column =  ['cont4','cont5','cont6','cont7',
                  'cont8','cont9','cont10','cont11',
                  'cont12']
for column in skewed:
    if column == 'cont1':
        all_data[column],lam = boxcox(all_data[column])
    elif column == 'cont13' or column == 'cont14':
        all_data[column] = np.abs(all_data[column] - np.mean(all_data[column]))
    elif column in boxcox_column:
        ## boxcox data must be positive
        all_data[column] = all_data[column] +1
        all_data[column],lam = boxcox(all_data[column])
        
all_data['cont2'] = np.tan(all_data['cont2'])


#df_dummies = pd.get_dummies(cat_list)
#print('Shape : ' ,all_data.shape)
#all_data.drop(cat_list,axis=1,inplace=True)
#print('Shape : ', all_data.shape)
#all_data = pd.concat((df_dummies,all_data),axis=1).reset_index(drop=True)
#print('Shape :',all_data.shape)
for cat in cat_list:
    all_data[cat] = pd.factorize(all_data[cat],sort=True)[0]


#all_data = all_data[cont_list].fillna(all_data[cont_list].mean())
scaler = StandardScaler().fit(all_data)
transform = scaler.transform(all_data)
x_train = scaler.transform(all_data.iloc[:num_trains])
x_test = scaler.transform(all_data.iloc[num_trains:])

#x_train = all_data.iloc[:num_trains]
#x_test = all_data.iloc[num_trains:]


#cv_params = {'gamma':[0,10e-1]}
ind_params = {'eta':0.1,
              'gamma':0.5290,
             # 'n_estimators':300,
              'max_depth':7,
              'min_child_weight':4.2922,
              'seed':42,
              'subsample':0.9930,
              'colsample_bytree':0.3085,
              'objective':'reg:linear',
              'silent':1}
#optimzation = GridSearchCV(xgb.XGBRegressor(**ind_params),
#                          cv_params,scoring='mean_squared_error',
#                          cv=5,n_jobs=-1)

#optimzation.fit(x_train,target)
#for params in optimzation.best_params_:
    
#    print("%s best parameter is %d "%(optimzation.best_params[params]))
#    ind_params[params] = optimzation.best_params[params]
x_test = xgb.DMatrix(x_test)
kf = KFold(x_train.shape[0],n_folds = 5)
for i,(train_idx,valid_idx) in enumerate(kf):
    print('Fold %d \n' %(i+1))
    
    X_train,X_val = x_train[train_idx],x_train[valid_idx]
    y_train,y_val = target[train_idx],target[valid_idx]
    
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dvalid = xgb.DMatrix(X_val,label=y_val)
    watchlist = [(dtrain,'train'),(dvalid,'eval')]
    
    clf = xgb.train(ind_params,dtrain,1000,watchlist,early_stopping_rounds = 25)
    
    score = clf.predict(dvalid,ntree_limit=clf.best_ntree_limit )
    cv_score = mean_absolute_error(np.exp(y_val),np.exp(score))
    print('MAE %0.4f' %(cv_score))
    y_pred = np.exp(clf.predict(x_test,
                    ntree_limit = clf.best_ntree_limit))
    
    if i>0:
        fpred = pred+y_pred
    else:
        fpred = y_pred
    
    pred = fpred
    
print('KFold Finished ....')
y_pred = fpred / 5
#print('Building the  Modle....')
#start_time = time.time()
#clf = xgb.XGBRegressor(**ind_params)
#clf.fit(x_train,target)
#y_pred = np.exp(clf.predict(x_test))
#print('Complete....')
#print('Time is %0.2f minutes' %((time.time() - start_time)/60))

now = datetime.datetime.now()
sub_file = 'Claims_Severity_submission_'  + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

pd.DataFrame({'id':test_ids,'loss':y_pred}).to_csv(sub_file,index=False)


    