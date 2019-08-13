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
# data split
df = dfoff[dfoff['label'] != -1].copy()
train = df[(df['Date_received'] < '20160516')].copy()
valid = df[(df['Date_received'] >= '20160516') & (df['Date_received'] <= '20160615')].copy()
print('Train Set: \n', train['label'].value_counts())
print('Valid Set: \n', valid['label'].value_counts())

#2.构建模型
def check_model(data, predictors):
   
   classifier = lambda: SGDClassifier(
       loss='log',  # loss function: logistic regression
       penalty='elasticnet', # L1 & L2
       fit_intercept=True,  # 是否存在截距，默认存在
       max_iter=100, 
       shuffle=True,  # Whether or not the training data should be shuffled after each epoch
       n_jobs=1, # The number of processors to use
       class_weight=None) # Weights associated with classes. If not given, all classes are supposed to have weight one.

   # 管道机制使得参数集在新数据集（比如测试集）上的重复使用，管道机制实现了对全部步骤的流式化封装和管理。
   model = Pipeline(steps=[
       ('ss', StandardScaler()), # transformer
       ('en', classifier())  # estimator
   ])

   parameters = {
       'en__alpha': [ 0.001, 0.01, 0.1],
       'en__l1_ratio': [ 0.001, 0.01, 0.1]
   }

   # StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
   folder = StratifiedKFold(n_splits=3, shuffle=True)
   
   # Exhaustive search over specified parameter values for an estimator.
   grid_search = GridSearchCV(
       model, 
       parameters, 
       cv=folder, 
       n_jobs=-1,  # -1 means using all processors
       verbose=1)
   grid_search = grid_search.fit(data[predictors], 
                                 data['label'])
   
   return grid_search

#3.训练   
predictors = original_feature
model = check_model(train, predictors)

#4.验证
# valid predict
y_valid_pred = model.predict_proba(valid[predictors])
valid1 = valid.copy()
valid1['pred_prob'] = y_valid_pred[:, 1]
valid1.head(5)

# avgAUC calculation
vg = valid1.groupby(['Coupon_id'])
aucs = []
for i in vg:
   tmpdf = i[1] 
   if len(tmpdf['label'].unique()) != 2:
       continue
   fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
   aucs.append(auc(fpr, tpr))
print(np.average(aucs))
