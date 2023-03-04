#importing libraries
import pandas as pd
import numpy as np

#reading the dataframe 
df=pd.read_excel('NASSCOM Final DATA.xlsx')

#converting Time column into time date format and setting as index
df.Time = pd.to_datetime(df.Time , format = '%Y-%m')
df.set_index(keys='Time',inplace=True)

#removing 'AAQ_PM2.5','AAQ_Ozone','AAQ_CO','AAQ_Benzene','Ind_Load','Nitrite-N (mg/L)','TFS (mg/L)' columns as they have more Nan values
df1=df.drop(columns=['AAQ_PM2.5','AAQ_Ozone','AAQ_CO','AAQ_Benzene','Ind_Load','Nitrite-N (mg/L)','TFS (mg/L)'])

#Interpolating missing values using time method
new_df=df1.interpolate(method='time')

#saving last 12 values for future prediction purpose
new_df[-12:].to_excel('test.xlsx')

#shifting  the values to 12 ahead so we can use the above 12 values(the last 12 values of the dataset) for future prediction from jan 2023 to dec 2023
col=new_df.columns
for i in col:
    new_df[i]=new_df[i].shift(12)

#Mapping correlation of features so best features can be used for exogeneous varoable in ARIMA
corr_matrix = new_df.corr()
print("-------------------------------------------------------Correlation-------------------------------------------------------")
print(corr_matrix)

#After several combinations the below are the features which aren't necessary i.e increase RMSE error value,so these are being dropped
col=col.drop(['AAQ_SO2','AQI', 'DO (mg/L)', 'pH','AAQ_SO2',
       'Conductivity (mS/cm)', 'BOD (mg/L)', 'Nitrate-N (mg/L)',
       'Turbidity (NTU)', 'Phen-Alk. (mg/L)', 'Total Alk. (mg/L)',
       'Chloride (mg/L)', 'COD (mg/L)', 'TKN (mg/L)',
       'Calcium Hardness (mg/l)', 'Magnesium (mg/L)', 'Sulphate (mg/L)',
       'Sodium (mg/L)', 'TDS (mg/L)', 'TSS (mg/L)', 'Phosphate (mg/L)',
       'Boron (mg/L)', 'Potassium (mg/L)', 'Fluoride (mg/L)', '%Sodium (mg/L)',
       'SAR (mg/L)','AAQ_NOx'])

#Train-Test Split
train=new_df[12:52]
test= new_df[52:]

#using auto_arima to find best order for the dataset
from pmdarima import auto_arima
stepwise_fit = auto_arima(train['AQI'], trace=True,suppress_warnings=True)
print(stepwise_fit)

#Training the ARIMA model using 'AQI' as endogenous data and 'AAQ_PM10', 'AAQ_NH3', 'Fecal Coliform (MPN/100ml)', 'Total Coliform (MPN/100ml)' as exogenous data
import statsmodels.api as sm
model=sm.tsa.arima.ARIMA(train['AQI'],order=(1,0,0),exog=train[col])
model=model.fit()
print("-------------------------------------------------------Model Summary-------------------------------------------------------")
print(model.summary())

#Generating Test data prediction values for model evaluation
start=40
end=47
import matplotlib.pyplot as plt
pred=model.predict(start=start,end=end,exog=test[col]).rename('ARIMA Predictions')
pred.plot(legend=True)
new_df['AQI'].plot(legend=True)

#Root mean square error is being used see the error betwwen true value and predicted
from sklearn.metrics import mean_squared_error
from math import sqrt
test['AQI'].mean()
rmse=sqrt(mean_squared_error(pred,test['AQI']))
print("-------------------------------------------------------RMSE Value-------------------------------------------------------")
print("RMSE value of the model: ",rmse)

#Reading the data stored at beginning for future prediction and using the values for exogenous variable in model.predict for better prediction
#If we generated random values for exogenous variable the model wouldn't have performed well on testing,to ensure accurate and better prediction values in the data are shifted i.e finding relation between present AQI values and previous exogenous value
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
print("-------------------------------------------------------Karimnagar AQI Prediction-------------------------------------------------------")
print("Date format: year-month-day")
print(pred[8:])
print("-------------------------------------------------------End-------------------------------------------------------")
