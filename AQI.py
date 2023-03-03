import pandas as pd
import numpy as np
df=pd.read_excel('NASSCOM Final DATA.xlsx')
df.Time = pd.to_datetime(df.Time , format = '%Y-%m')
df.set_index(keys='Time',inplace=True)
df1=df.drop(columns=['AAQ_PM2.5','AAQ_Ozone','AAQ_CO','AAQ_Benzene','Ind_Load','Nitrite-N (mg/L)','TFS (mg/L)'])
new_df=df1.interpolate(method='time')
new_df[-12:].to_excel('test.xlsx')
col=new_df.columns
for i in col:
    new_df[i]=new_df[i].shift(12)
new_df
corr_matrix = new_df.corr()
corr_matrix
col=col.drop(['AAQ_SO2','AQI', 'DO (mg/L)', 'pH','AAQ_SO2',
       'Conductivity (mS/cm)', 'BOD (mg/L)', 'Nitrate-N (mg/L)',
       'Turbidity (NTU)', 'Phen-Alk. (mg/L)', 'Total Alk. (mg/L)',
       'Chloride (mg/L)', 'COD (mg/L)', 'TKN (mg/L)',
       'Calcium Hardness (mg/l)', 'Magnesium (mg/L)', 'Sulphate (mg/L)',
       'Sodium (mg/L)', 'TDS (mg/L)', 'TSS (mg/L)', 'Phosphate (mg/L)',
       'Boron (mg/L)', 'Potassium (mg/L)', 'Fluoride (mg/L)', '%Sodium (mg/L)',
       'SAR (mg/L)','AAQ_NOx'])
train=new_df[12:52]
test= new_df[52:]
from pmdarima import auto_arima
stepwise_fit = auto_arima(train['AQI'], trace=True,suppress_warnings=True)
print(stepwise_fit)
import statsmodels.api as sm
model=sm.tsa.arima.ARIMA(train['AQI'],order=(1,0,0),exog=train[col])
model=model.fit()
print("-------------------------------------------------------Model Summary-------------------------------------------------------")
print(model.summary())
start=40
end=47
import matplotlib.pyplot as plt
pred=model.predict(start=start,end=end,exog=test[col]).rename('ARIMA Predictions')
pred.plot(legend=True)
new_df['AQI'].plot(legend=True)
from sklearn.metrics import mean_squared_error
from math import sqrt
test['AQI'].mean()
rmse=sqrt(mean_squared_error(pred,test['AQI']))
print("-------------------------------------------------------RMSE Value-------------------------------------------------------")
print("RMSE value of the model: ",rmse)
data=pd.read_excel('test.xlsx')
data1=data[col]
data1.to_excel('test.xlsx')
data1=pd.read_excel('test.xlsx')
data1.drop(columns=data1.columns[0],inplace=True)
final_df = pd.concat([test[col],data1])
start=40
end=59
pred=model.predict(start=start,end=end,exog=final_df).rename('ARIMA Predictions')
pred.plot(legend=True)
new_df['AQI'].plot(legend=True)
print("-------------------------------------------------------Prediction-------------------------------------------------------")
print(pred)
print("-------------------------------------------------------End-------------------------------------------------------")