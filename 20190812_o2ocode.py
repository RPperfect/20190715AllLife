import pandas as pd
import numpy as np
from datetime import date
#加载线下数据
dfoff=pd.read_csv('F:\IT\阿里天池\天池新人实战赛o2o优惠\data\ccf_offline_stage1_train.csv')
dftest=pd.read_csv('F:\IT\阿里天池\天池新人实战赛o2o优惠\data\ccf_offline_stage1_test_revised.csv')

#数据前5行
dfoff.head()

#统计用户是否使用优惠券消费的情况
print('有优惠卷，购买商品：%d' % dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] != 'null')].shape[0])
print('有优惠卷，未购商品：%d' % dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] == 'null')].shape[0])
print('无优惠卷，购买商品：%d' % dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] != 'null')].shape[0])
print('无优惠卷，未购商品：%d' % dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] == 'null')].shape[0])

#特征提取
#1.打折率(Discount_rate)
print('Discount_rate 类型：\n',dfoff['Discount_rate'].unique())
print('Discount_rate 类型：\n',set(dfoff['Discount_rate']))

#打折类型 3种，null表示没有打折 [0,1]折扣率 x:y满x减y
def getDiscountType(row):
   if row == 'null':
       return 'null'
   elif ':' in row:
       return 1
   else:
       return 0

#折扣率 统一转换为[0,1]折扣率
def convertRate(row):
   """Convert discount to rate"""
   if row == 'null':
       return 1.0
   elif ':' in row:
       rows = row.split(':')
       return 1.0 - float(rows[1])/float(rows[0])
   else:
       return float(row)

#满多少
def getDiscountMan(row):
   if ':' in row:
       rows = row.split(':')
       return int(rows[0])
   else:
       return 0
#减多少
def getDiscountJian(row):
   if ':' in row:
       rows = row.split(':')
       return int(rows[1])
   else:
       return 0
       
def processData(df):   
   # convert discount_rate
   df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
   df['discount_rate'] = df['Discount_rate'].apply(convertRate)
   df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
   df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
   print(df['discount_rate'].unique())
   return df
   
#对训练集和测试集分别进行进行 processData（）函数的处理
dfoff=processData(dfoff)
dftest=processData(dftest)

#2.距离(Distance)
print('Distance 类型：',dfoff['Distance'].unique())
# convert distance
dfoff['distance'] = dfoff['Distance'].replace('null', -1).astype(int)
print(dfoff['distance'].unique())
dftest['distance'] = dftest['Distance'].replace('null', -1).astype(int)
print(dftest['distance'].unique())

#3.领劵日期（Date_received）
def getWeekday(row):
   if row == 'null':
       return row
   else:
       return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1

dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)

# weekday_type :  周六和周日为1，其他为0
dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x: 1 if x in [6,7] else 0)
dftest['weekday_type'] = dftest['weekday'].apply(lambda x: 1 if x in [6,7] else 0)

# change weekday to one-hot encoding 
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
#print(weekdaycols)

tmpdf = pd.get_dummies(dfoff['weekday'].replace('null', np.nan))
tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf

tmpdf = pd.get_dummies(dftest['weekday'].replace('null', np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf

#经过特征提取，增加了14个特征

#标注标签label 正样本y=1,负样本y=0
def label(row):
   if row['Date_received'] == 'null':
       return -1
   if row['Date'] != 'null':
       td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
       if td <= pd.Timedelta(15, 'D'):
           return 1
   return 0

dfoff['label']=dfoff.apply(label,axis=1)
print(dfoff['label'].value_counts())

#建立模型
#1.划分训练集和验证集 划分方式是按照领券日期，即训练集：20160101-20160515，验证集：20160516-20160615。我们采用的模型是简单的 SGDClassifier。
