# Machine_Learning_one 线性模型
import numpy as np
import pandas as pd
path='datas/household_power_consumption_1000.txt'
df = pd.read_csv(path,sep=";") # 使用pandas加载csv

## 对数据集增加偏置，并去除掉不需要的列
b = pd.DataFrame([1] * df.shape[0],columns=["b"])  # 建立一列为1的dataframe，起名为b作为偏置项
df = pd.concat([df,b],axis=1)   # 将偏置项合并到原始dataframe
df["b"]=1
## 获取Global_active_power、Global_reactive_power、Global_intensity和偏置列b的dataframe
df = df[["Global_active_power","Global_reactive_power","b","Global_intensity"]]

## 异常数据处理
df = df.replace("?",np.nan).dropna() # 只要有特征为空，就进行删除操作
df["Global_active_power"] = df["Global_active_power"].astype(np.float64)#将字符串转为浮点

## 分离X和Y
## 获取"Global_active_power","Global_reactive_power","b"为X
X = df.iloc[:,:-1]
Y = df.iloc[:,[-1]] #获取Global_intensity作为Y

## 将dataframe转为numpy矩阵
X = np.mat(X) #将X转为numpy矩阵
Y = np.mat(Y)  #将Y转为numpy矩阵

## 使用正规方程法求解模型参数
(X.T*X).I*X.T*Y

————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
import numpy as np
import pandas as pd
path='datas/household_power_consumption_1000.txt'
df = pd.read_csv(path,sep=";")   # 使用pandas加载csv

## 对数据集增加偏置，并去除掉不需要的列
b = pd.DataFrame([1] * df.shape[0],columns=["b"])   # 建立一列为1的dataframe，起名为b作为偏置项
df = pd.concat([df,b],axis=1)    # 将偏置项合并到原始dataframe
df["b"]=1
## 获取Global_active_power、Global_reactive_power、Global_intensity和偏置列b的dataframe
df = df[["Global_active_power","Global_reactive_power","b","Global_intensity"]]

## 异常数据处理
df = df.replace("?",np.nan).dropna()    # 只要有特征为空，就进行删除操作
df["Global_active_power"] = df["Global_active_power"].astype(np.float64)   # 将字符串转为浮点

## 分离X和Y
## 获取"Global_active_power","Global_reactive_power","b"为X
X = df.iloc[:,:-1]
Y = df.iloc[:,[-1]]      # 获取Global_intensity作为Y

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=False)
model.fit(X,Y)
model.coef_
model.score(X,Y)
