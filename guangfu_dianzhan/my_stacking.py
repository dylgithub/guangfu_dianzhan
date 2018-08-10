#encoding=utf-8
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,explained_variance_score
import numpy as np
import xgboost as gbt
from guangfu_dianzhan import data_process
train_df, test_df, train_label = data_process.get_person_data()
train_df.drop('ID', axis=1, inplace=True)
train_data = train_df.values
id_list=list(test_df.pop('ID'))
test_data=test_df.values
X_train,X_test,y_train,y_test=train_test_split(train_data,train_label,test_size=0.25,random_state=3)
N_Folds=5
ntrain=X_train.shape[0]
ntest=X_test.shape[0]
kf=KFold(n_splits=N_Folds,random_state=5)
def get_off(clf,X_train,y_train,X_test):
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_test=np.array(X_test)
    off_train=np.zeros((ntrain,))
    off_test=np.zeros((ntest,))
    off_test_skf=np.zeros((N_Folds,ntest))
    for i,(train_index,test_index) in enumerate(kf.split(X_train)):
        print(train_index)
        kf_X_train=X_train[train_index]
        kf_y_train=y_train[train_index]
        kf_X_test=X_train[test_index]
        clf.fit(kf_X_train,kf_y_train)
        off_train[test_index]=clf.predict(kf_X_test)
        off_test_skf[i,:]=clf.predict(X_test)
    off_test[:]=off_test_skf.mean(axis=0)
    return off_train.reshape(-1,1),off_test.reshape(-1,1)
clf1 = gbt.XGBRegressor(n_estimators=1000, subsample=0.8, learning_rate=0.25, objective='reg:linear')
clf2 = gbt.XGBRegressor(n_estimators=1000, subsample=0.8, learning_rate=0.25, objective='reg:gamma')
clf3 = gbt.XGBRegressor(n_estimators=1000, subsample=0.8, learning_rate=0.25, objective='reg:tweedie')
# for clf in [clf1,clf2]:
#     clf.fit(X_train,y_train)
#     yHat=clf.predict(X_test)
#     print(mean_squared_error(y_test, yHat))
#     print(explained_variance_score(y_test, yHat))
train_data,test_data=get_off(clf1,X_train,y_train,X_test)
train_data2,test_data2=get_off(clf2,X_train,y_train,X_test)


#这里不是树形模型
train_data_scale, test_data_scale,_=data_process.get_scale_data()
X_train_scale,X_test_scale,y_train_scale,y_test_scale=train_test_split(train_data_scale,test_data_scale,test_size=0.25,random_state=3)

data=np.concatenate((train_data,train_data2),axis=1)
test=np.concatenate((test_data,test_data2),axis=1)
clf3.fit(data,y_train)
yHat=clf3.predict(test)
print(mean_squared_error(y_test, yHat))
print(explained_variance_score(y_test, yHat))