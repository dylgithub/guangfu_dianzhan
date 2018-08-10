#encoding=utf-8
import xgboost as gbt
import numpy as np
import pandas as pd
from guangfu_dianzhan import data_process
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.metrics import mean_squared_error,explained_variance_score
'''
模型融合效果会有提高，十折交叉取均值效果会有提高
加入XGBoost提取的特征效果并没有有效提高
加入XGBoost提取的特征进行模型融合再取十折交叉效果没有直接对数据进行模型融合再取十折交叉效果好
'''
def train_model():
    train_df, _, train_label=data_process.get_person_data()
    train_df.drop('ID',axis=1,inplace=True)
    train_data=train_df.values
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2,random_state=66)
    model = gbt.XGBRegressor(n_estimators=1000, subsample=0.8, learning_rate=0.25, objective='reg:tweedie')
    # model = gbt.XGBRegressor(n_estimators=1000, subsample=0.8, learning_rate=0.25, objective='reg:linear')
    # model = gbt.XGBRegressor(n_estimators=1000, subsample=0.8, learning_rate=0.25, objective='reg:gamma')
    # model=AdaBoostRegressor(n_estimators=200)
    model.fit(X_train,y_train)
    new_X_train = model.apply(X_train)
    new_X_val = model.apply(X_val)
    model.fit(new_X_train, y_train)
    yHat=model.predict(new_X_val)
    print(mean_squared_error(y_val,yHat))
    print(explained_variance_score(y_val,yHat))
def model_run():
    train_df, test_df, train_label = data_process.get_data()
    train_df.drop('ID', axis=1, inplace=True)
    train_data = train_df.values
    id_list=list(test_df.pop('ID'))
    test_data=test_df.values
    model = gbt.XGBRegressor(n_estimators=1000, subsample=0.8, learning_rate=0.25, objective='reg:tweedie')
    model.fit(train_data, train_label)
    # yHat=model.predict(test_data)
    # result=pd.DataFrame({
    #     'id':id_list,
    #     'yhat':yHat
    # })
    # result.to_csv('result/result1.csv',index=False,header=None,encoding='utf-8')
def model_stack():
    kf = KFold(n_splits=10, random_state=5)
    train_df, test_df, train_label = data_process.get_person_data()
    train_df.drop('ID', axis=1, inplace=True)
    train_data = train_df.values
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=66)
    id_list=list(test_df.pop('ID'))
    test_data=test_df.values
    model1 = gbt.XGBRegressor(n_estimators=1000, subsample=0.8, learning_rate=0.25, objective='reg:linear')
    model2 = gbt.XGBRegressor(n_estimators=1000, subsample=0.8, learning_rate=0.25, objective='reg:gamma')
    model3 = gbt.XGBRegressor(n_estimators=1000, subsample=0.8, learning_rate=0.25, objective='reg:tweedie')
    model4 = RandomForestRegressor()
    model5 = GradientBoostingRegressor()
    # model6 = AdaBoostRegressor(n_estimators=200)
    stack_model = StackingClassifier(classifiers=[model1, model2,model4,model5],
                              meta_classifier=model3)
    train_data=np.array(train_data)
    yHat_list=[]
    for i, (train_index, test_index) in enumerate(kf.split(train_data)):
        new_train_data=train_data[train_index]
        new_train_label=[train_label[i] for i in train_index]
        stack_model.fit(new_train_data, new_train_label)
        yHat = stack_model.predict(test_data)
        yHat_list.append(yHat)
    yHat_list=np.array(yHat_list)
    yHat_list=yHat_list.mean(axis=0)
    result = pd.DataFrame({
        'id': id_list,
        'yhat': yHat_list
    })
    result.to_csv('result/result17.csv', index=False, header=None, encoding='utf-8')
    # print(mean_squared_error(y_val, yHat))
    # print(explained_variance_score(y_val, yHat))
def model_stack2():
    _, test_df, train_label = data_process.get_person_data()
    train_data, test_data = data_process.get_scale_data()
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=66)
    id_list=list(test_df.pop('ID'))
    model1 = gbt.XGBRegressor(n_estimators=1000, subsample=0.8, learning_rate=0.25, objective='reg:linear')
    model2 = gbt.XGBRegressor(n_estimators=1000, subsample=0.8, learning_rate=0.25, objective='reg:gamma')
    model3 = gbt.XGBRegressor(n_estimators=1000, subsample=0.8, learning_rate=0.25, objective='reg:tweedie')
    model4 = svm.SVR()
    stack_model = StackingClassifier(classifiers=[model1, model2,model3,model4],
                              meta_classifier=model3)
    stack_model.fit(train_data, train_label)
    yHat = stack_model.predict(test_data)
    result = pd.DataFrame({
        'id': id_list,
        'yhat': yHat
    })
    result.to_csv('result/result6.csv', index=False, header=None, encoding='utf-8')
    # print(mean_squared_error(y_val, yHat))
    # print(explained_variance_score(y_val, yHat))
def train_svr():
    train_data, test_data,train_label=data_process.get_scale_data()
    _, test_df,_ = data_process.get_person_data()
    id_list = list(test_df.pop('ID'))
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=66)
    model2=svm.SVR()
    # model2 = gbt.XGBRegressor(n_estimators=1000, subsample=0.8, learning_rate=0.25, objective='reg:tweedie')
    model2.fit(train_data, train_label)
    yHat = model2.predict(test_data)
    result = pd.DataFrame({
        'id': id_list,
        'yhat': yHat
    })
    result.to_csv('result/result11.csv', index=False, header=None, encoding='utf-8')
    # print(mean_squared_error(y_val, yHat))
    # print(explained_variance_score(y_val, yHat))
if __name__ == '__main__':
    model_stack()