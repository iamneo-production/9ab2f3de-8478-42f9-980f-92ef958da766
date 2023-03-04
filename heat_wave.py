import pandas as pd
import numpy as np
df= pd.read_excel('nassc_weather.xlsx')
df.date = pd.to_datetime(df.date , format = '%Y-%m')
df.set_index(keys='date',inplace=True)
new_df=df.interpolate(method='time')
new_df[-365:].to_excel('test2.xlsx')
col=new_df.columns
for i in col:
    new_df[i]=new_df[i].shift(365)
corr_matrix = new_df.corr()
print(corr_matrix)
col=col.drop(['temp_max ','wind_speed_min','AQI','wind_speed_max ','rainfall'])
train=new_df[365:1517]
test= new_df[1517:]
from pmdarima import auto_arima
stepwise_fit = auto_arima(train['temp_max '], trace=True,suppress_warnings=True)
print(stepwise_fit)
import statsmodels.api as sm
model=sm.tsa.arima.ARIMA(train['temp_max '],order=(1,1,2),exog=train[col])
model=model.fit()
print("-------------------------------------------------------Model Summary-------------------------------------------------------")
print(model.summary())
start=1152
end=1439
pred=model.predict(start=start,end=end,exog=test[col]).rename('ARIMA Predictions')
from datetime import datetime
date=pd.date_range(start="2022-03-19",end="2022-12-31")
pred = pd.DataFrame(data=pred)
pred.set_index(keys=[date],inplace=True)
pred.plot(legend=True)
new_df['temp_max '].plot(legend=True)
from sklearn.metrics import mean_squared_error
from math import sqrt
test['temp_max '].mean()
rmse=sqrt(mean_squared_error(pred,test['temp_max ']))
print("-------------------------------------------------------Metric Evaluation-------------------------------------------------------")
print(rmse)
test['heat_wave']= test['temp_max '].apply(lambda x: 1 if x>40 else 0)
pred['heat_wave']= pred['ARIMA Predictions'].apply(lambda x: 1 if x>38.8 else 0)
from sklearn.metrics import roc_auc_score,confusion_matrix
import matplotlib.pyplot as plt
roc_auc = roc_auc_score(test['heat_wave'], pred['heat_wave'])
cm = confusion_matrix(test['heat_wave'], pred['heat_wave'])
print("AUC-ROC score:", roc_auc)
print("Confusion Matrix: ",cm)
data=pd.read_excel('test2.xlsx')
data1=data[col]
final_df = pd.concat([test[col],data1])
start=1152
end=1804
pred=model.predict(start=start,end=end,exog=final_df).rename('ARIMA Predictions')
date=pd.date_range(start="2022-03-19",end="2023-12-31")
pred = pd.DataFrame(data=pred)
pred.set_index(keys=date,inplace=True)
pred.plot(legend=True)
new_df['temp_max '].plot(legend=True)
print("-------------------------------------------------------Prediction-------------------------------------------------------")
print(pred)
print("-------------------------------------------------------Heat Wave Occurence in karimnagar-------------------------------------------------------")
pred['heat_wave']= pred['ARIMA Predictions'].apply(lambda x: 1 if x>38.8 else 0)
print(pred[pred['heat_wave']==1])
print("-------------------------------------------------------End-------------------------------------------------------")

