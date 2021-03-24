import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt



data = []
target = []
with open('刀具磨耗_24筆.csv') as f:     ##將資料撈取
    mycsv = csv.reader(f)
    headers = next(mycsv)
    for row in mycsv:
        data.append(list(map(float,row[0:4])))
        target.append(float(row[4]))

data = np.array(data)               ##轉array
data = data.reshape(-1,4)
target = np.array(target)
target = target.reshape(-1,1)
print(data[2])

ss = StandardScaler()               ##正規化

data = ss.fit_transform(data)
#target = ss.fit_transform(target)

##print(data.shape[0])
##print(len(data))

##data = np.array(data)
##data = data.reshape(-1,4)
##target = np.array(target)
##target = target.reshape(-1,1)


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.8, random_state=0)    ##建立訓練集大小

regressor = LinearRegression()  
regressor.fit(X_train, y_train)     ##訓練模型

print('Intercept:{}'.format(regressor.intercept_))
print('Coefficient:{}'.format(regressor.coef_),'\n')


y_pred = regressor.predict(X_test)  ##預測測試集 & 計算參數
print(y_pred)
print(y_test)
#print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
#print('Variance score: %.2f' % r2_score(y_test, y_pred))

mse = np.sum((y_pred - y_test)**2)
rmse = np.sqrt(mse/len(data))
rms = sqrt(mean_squared_error(y_test, y_pred))

print('RMSE:',rmse)
##print('RMS:',rms)

ssr = np.sum((y_pred - y_test)**2)
sst = np.sum((y_test - np.mean(y_test))**2)
r2_score = 1 - (ssr/sst)
print('SSR:',r2_score,'\n')


##print('Difference:',y_test-y_pred)
##print(y_pred)
##print(invers_tainsform(y_pred))
##print(y_test)

plt.figure()                                                                        ##畫圖
##plt.subplot(2,1,1)
##plt.plot(range(len(y_pred)),ss.inverse_transform(y_pred),'b',label="predict")
##plt.plot(range(len(y_pred)),ss.inverse_transform(y_test),'r',label="test")
##plt.legend(loc="upper right") #显示图中的标签
##plt.xlabel('')
##plt.ylabel('')

##plt.subplot(2,1,2)
##plt.plot(range(len(y_pred)),ss.inverse_transform(y_test-y_pred),'y',label="Difference")
##plt.legend(loc="upper right") #显示图中的标签
##plt.xlabel('')
##plt.ylabel('')

##x = []
##for i in range(24):
##    x.append(i)
##
##x = np.array(x)
##x = x.reshape(-1,1)
##
##plt.subplot(2,2,3)
##plt.scatter(x , y_test , s=10)
##plt.xlabel('')
##plt.ylabel('')





#=================預測刀具磨耗_2===================================
##data_2 = []
##target_2 = []
##with open('刀具磨耗_2.csv') as f:
##    mycsv = csv.reader(f)
##    headers = next(mycsv)
##    for row in mycsv:
##        data_2.append(row[0:4])
##        target_2.append(row[4])
##
##
##data_2 = np.array(data_2)
##data_2 = data_2.reshape(-1,4)
##target_2 = np.array(target)
##target_2 = target_2.reshape(-1,1)
##data_2 = ss.fit_transform(data_2)
##target_2 = ss.fit_transform(target_2)
##
##y_pred_2 = regressor.predict(data_2)
##print(' 8筆資料預測:\n',ss.inverse_transform(y_pred_2))



data_3 = []
target_3 = []
with open('刀具磨耗_轉速.csv') as f:
    mycsv = csv.reader(f)
    headers = next(mycsv)
    for row in mycsv:
        data_3.append(row[0:4])
        target_3.append(row[4])


data_3 = np.array(data_3)
data_3 = data_3.reshape(-1,4)
data_3 = ss.fit_transform(data_3)
target_3 = np.array(target)
target_3 = target_3.reshape(-1,1)
target_3 = ss.fit_transform(target_3)


y_pred_3 = regressor.predict(data_3)

print(' 8筆資料預測:\n',ss.inverse_transform(y_pred_3))

                                                                      ##畫圖
##plt.subplot(2,1,1)

x = []
for i in range(10):
    a = i * 1000 + 1000
    x.append(a)
x = np.array(x)
x = x.reshape(-1,1)
plt.plot(x,ss.inverse_transform(y_pred_3),'b',label="predict")
plt.legend(loc="upper right") #显示图中的标签
plt.xlabel('Rpm')
plt.ylabel('Predict')



plt.show()



























