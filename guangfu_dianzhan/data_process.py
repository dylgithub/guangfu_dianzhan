#encoding=utf-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,scale
def get_sum_mean(df):
    df['转换效率sum']=df['转换效率A']+df['转换效率B']+df['转换效率C']
    df['电压sum']=df['电压A']+df['电压B']+df['电压C']
    df['电压mean']=(df['电压A']+df['电压B']+df['电压C'])/3.0
    df['电流sum']=df['电流A']+df['电流B']+df['电流C']
    df['电流mean']=(df['电流A']+df['电流B']+df['电流C'])/3.0
    df['功率sum']=df['功率A']+df['功率B']+df['功率C']
    return df
def get_var(df):
    df['转换效率var']=df[['转换效率A','转换效率B','转换效率C']].var(1)
    df['电压var']=df[['电压A','电压B','电压C']].var(1)
    df['电流var']=df[['电流A','电流B','电流C']].var(1)
    df['功率var']=df[['功率A','功率B','功率C']].var(1)
    return df
def get_lisan(df):
    df['风速']=pd.cut(df['风速'],10,labels=range(10))
    df['光照强度']=pd.cut(df['光照强度'],10,labels=range(10))
    return df
def get_he_cha(df):
    wencha=[]
    # wenhe=[]
    for i,j in zip(list(df['板温']),list(df['现场温度'])):
        wencha.append(np.abs(np.abs(i)-np.abs(j)))
    df['温差']=wencha
    return df
def get_data():
    train_df=pd.read_csv('data/public.train.csv')
    test_df=pd.read_csv('data/public.test.csv')
    train_df=train_df[train_df['发电量']>=0]
    train_label=list(train_df.pop('发电量'))
    # datas=pd.concat([train_df,test_df])
    # print(train_label)
    # print(datas.head())
    return train_df,test_df,train_label
def get_person_data():
    train_df = pd.read_csv('data/public.train.csv')
    test_df = pd.read_csv('data/public.test.csv')
    train_df = train_df[train_df['发电量'] >= 0]
    test_df['发电量']=666
    datas=pd.concat([train_df,test_df],axis=0)
    datas=get_sum_mean(datas)
    datas=get_he_cha(datas)
    # datas=get_lisan(datas)
    # datas=get_var(datas)
    new_train_df=datas[datas['发电量']!=666]
    new_test_df=datas[datas['发电量']==666]
    train_label = list(new_train_df.pop('发电量'))
    test_label = list(new_test_df.pop('发电量'))
    # new_test_df.drop('发电量',axis=1,inplace=True)
    return new_train_df, new_test_df, train_label
#把数据进行最小最大值归一化之后验证集效果很好但是提交之后的测试集效果特别差
def get_scale_data():
    new_train_df, new_test_df, train_label=get_person_data()
    train_id = list(new_train_df.pop('ID'))
    test_id = list(new_test_df.pop('ID'))
    train_data=new_train_df.values
    test_data=new_test_df.values
    # scaler=MinMaxScaler()
    # train_data=scaler.fit_transform(train_data)
    # test_data=scaler.fit_transform(test_data)
    train_data=scale(train_data)
    test_data=scale(test_data)
    # print(train_data.std(axis=1))  # 标准差为1
    # train_label=scaler2.fit_transform(train_label)
    return train_data,test_data,train_label
if __name__ == '__main__':
    new_train_df, new_test_df, train_label=get_person_data()
    print(new_train_df['风速'].max())
    print(new_train_df['风速'].min())