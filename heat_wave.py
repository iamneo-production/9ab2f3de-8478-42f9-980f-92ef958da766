#importing libraries
import pandas as pd
import numpy as np

#reading the dataframe 
df= pd.read_excel('nassc_weather.xlsx')

#converting Time column into time date format and setting as index
df.date = pd.to_datetime(df.date , format = '%Y-%m')
df.set_index(keys='date',inplace=True)

#Interpolating missing values using time method,so the data will be clean as clean data need ti fed to train a model
new_df=df.interpolate(method='time')

#saving last 365 values for future prediction purpose i.e prediction 2023 
new_df[-365:].to_excel('test2.xlsx')

#shifting  the values to 365 ahead so we can use the above 365 values(the last 365 values of the dataset) for future prediction from jan 2023 to dec 2023 which is day wise
col=new_df.columns
for i in col:
    new_df[i]=new_df[i].shift(365)

#mapping correlation to find best fetures for the model
corr_matrix = new_df.corr()
print(corr_matrix)

#After several combinations the below are the features which aren't necessary i.e increase RMSE error value,so these are being dropped
col=col.drop(['temp_max ','wind_speed_min','AQI','wind_speed_max ','rainfall'])

#Test-Train split of the data
train=new_df[365:1517]
test= new_df[1517:]

#using auto_arima to find best order for the dataset
from pmdarima import auto_arima
stepwise_fit = auto_arima(train['temp_max '], trace=True,suppress_warnings=True)
print(stepwise_fit)

#Training the ARIMA model using 'temp_max' as endogenous data and 'temp_min ', 'temp_max ', 'humidity_min', 'humidity_max 'as exogenous data
import statsmodels.api as sm
model=sm.tsa.arima.ARIMA(train['temp_max '],order=(1,1,2),exog=train[col])
model=model.fit()
print("-------------------------------------------------------Model Summary-------------------------------------------------------")
print(model.summary())
print("------------------------------------------------------------------------------------------------------------------------")

#Generating Test data prediction values for model evaluation
start=1152
end=1439
pred=model.predict(start=start,end=end,exog=test[col]).rename('ARIMA Predictions')
from datetime import datetime
date=pd.date_range(start="2022-03-19",end="2022-12-31")#preparing time index for the predicted value
pred = pd.DataFrame(data=pred)
pred.set_index(keys=[date],inplace=True)
import matplotlib.pyplot as plt
pred.plot(legend=True)
new_df['temp_max '].plot(legend=True)

#Root mean square error is being used see the error betwwen true value and predicted
from sklearn.metrics import mean_squared_error
from math import sqrt
test['temp_max '].mean()
rmse=sqrt(mean_squared_error(pred,test['temp_max ']))

#If the temperature is more than 40 then it is considered as Heatwave,so we will be marking the row with temp_max>40 as Heatwave(1 is used to indicate it)
test['heat_wave']= test['temp_max '].apply(lambda x: 1 if x>40 else 0)
#In the case of predicted we are taking 38.8 as due to error of some 2.09 the value predicted might be a bit more or less
#so a range of values between 38-42 is used 
#After several values,38.8 is the value where we get a good prediction(AUC_ROC score and confusion matrix),so we considered 38.8 as value for the predicted ones to classify as heatwave occurred or not.
pred['heat_wave']= pred['ARIMA Predictions'].apply(lambda x: 1 if x>38.8 else 0)

#Metric evaluation
from sklearn.metrics import roc_auc_score,confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
roc_auc = roc_auc_score(test['heat_wave'], pred['heat_wave'])
cm = confusion_matrix(test['heat_wave'], pred['heat_wave'])
acc=accuracy_score(test['heat_wave'], pred['heat_wave'])
print("-------------------------------------------------------Metric Evaluation-------------------------------------------------------")
print("RMSE score:",rmse)
print("Accuracy score:", acc)
print("AUC-ROC score:", roc_auc)
print("Confusion Matrix: ",cm)
print("------------------------------------------------------------------------------------------------------------------------")

##Reading the data stored at beginning for future prediction and using the values for exogenous variable in model.predict for better prediction
#If we generated random values for exogenous variable the model wouldn't have performed well on testing,to ensure accurate and better prediction values in the data are shifted i.e finding relation between present temp_max values and previous exogenous value
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
print("------------------------------------------------------------------------------------------------------------------------")
pred['heat_wave']= pred['ARIMA Predictions'].apply(lambda x: 1 if x>38.8 else 0)#as said 38.8 is used as a base line to classify heatwave for predicted value
heat_wave_occurence=pred[pred['heat_wave']==1]
heat_wave_occurence.reset_index(inplace=True)
print("---------------------------------Dates for Heat Wave Occurence in karimnagar--------------------------------------------")
print("Date format: year-month-day")
print(heat_wave_occurence['index'][26:])
print("-------------------------------------------------------End--------------------------------------------------------------")

