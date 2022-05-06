# Regression-Accuracies
import pandas as pd 
 
df = pd.read_csv('finalised_dataset.csv',na_values='=') 
 
df=df.drop('Yield', axis = 1) 
 
df.info() 
 
df.columns 
 
df = df[df['state_names'] == "Maharashtra"] 
 
  
 
df.info() 
 
df.info() 
 
df.isnull().sum() 
 
df.head(6) 
 
import matplotlib.pyplot as plt 
 
import seaborn as sb 
 
  
 
C_mat = df.corr() 
 
fig = plt.figure(figsize = (15,15)) 
 
  
 
sb.heatmap(C_mat, vmax = .8, square = True) 
 
plt.show() 
 
df = df[df['crop_year']>=2004 
 
df.info() 
 
df = df.join(pd.get_dummies(df['district_names'])) 
 
df = df.join(pd.get_dummies(df['season_names'])) 
 
df = df.join(pd.get_dummies(df['crop_names'])) 
 
df = df.join(pd.get_dummies(df['state_names'])) 
 
df = df.join(pd.get_dummies(df['soil_type'])) 
 
df['Yield'] = df['production']/df['area'] 
 
df = df.drop('production', axis=1) df = df.drop('production', axis=1) 
 
df=df.drop('district_names', axis=1) 
 
df = df.drop('season_names',axis=1) 
 
df = df.drop('crop_names',axis=1) 
 
 
 
df = df.drop('state_names', axis=1) 
 
df = df.drop('soil_type', axis=1) 
 
from sklearn import preprocessing 
 
# Create x, where x the 'scores' column's values as floats 
 
x = df[['area']].values.astype(float) 
 
x 
 
# Create a minimum and maximum processor object 
 
min_max_scaler = preprocessing.MinMaxScaler() 
 
  
 
# Create an object to transform the data to fit minmax processor 
 
x_scaled = min_max_scaler.fit_transform(x) 
 
  
 
# Run the normalizer on the dataframe 
 
#df_normalized = pd.DataFrame(x_scaled) 
 
x_scaled 
 
  
 
df['area'] = x_scaled 
 
Df 
 
df.head() 
 
df = df.fillna(df.mean()) 
 
from sklearn.model_selection import train_test_split 
 
a=df 
 
b = df['Yield'] 
 
#a = df.drop('Yield', axis = 1) 
 
 
 
c = df.drop('Unnamed: 0', axis = 1) 
 
a=c.drop('Yield', axis = 1) 
 
len(a.columns) 
 
a.columns 
 
features_list=['crop_year', 'area', 'temperature', 'wind_speed', 'pressure', 
 
       'humidity', 'N', 'P', 'K', 'AHMEDNAGAR', 'AKOLA', 'AMRAVATI', 
 
       'AURANGABAD', 'BEED', 'BHANDARA', 'BULDHANA', 'CHANDRAPUR', 'DHULE', 
 
       'GADCHIROLI', 'GONDIA', 'HINGOLI', 'JALGAON', 'JALNA', 'KOLHAPUR', 
 
       'LATUR', 'NAGPUR', 'NANDED', 'NANDURBAR', 'NASHIK', 'OSMANABAD', 
 
       'PALGHAR', 'PARBHANI', 'PUNE', 'RAIGAD', 'RATNAGIRI', 'SANGLI', 
 
       'SATARA', 'SINDHUDURG', 'SOLAPUR', 'THANE', 'WARDHA', 'WASHIM', 
 
       'YAVATMAL', 'Kharif     ', 'Rabi       ', 'Summer     ', 'Whole Year ', 
 
       'Arhar/Tur', 'Bajra', 'Castor seed', 'Cotton(lint)', 'Gram', 
 
       'Groundnut', 'Jowar', 'Linseed', 'Maize', 'Moong(Green Gram)', 
 
       'Niger seed', 'Other  Rabi pulses', 'Other Cereals & Millets', 
 
       'Other Kharif pulses', 'Ragi', 'Rapeseed &Mustard', 'Rice', 'Safflower', 
 
       'Sesamum', 'Soyabean', 'Sugarcane', 'Sunflower', 'Tobacco', 'Urad', 
 
       'Wheat', 'other oilseeds', 'Maharashtra', 'chalky', 'clay', 'loamy', 
 
       'peaty', 'sandy', 'silt', 'silty'] 
 
features_list123=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
 
  
 
len(features_list123) 
 
len(features_list) 
 
a=df[features_list] 
 
a.head() 
 
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.3, random_state = 42) 
 
  
 
print(a_train) 
 
print(a_test) 
 
print(b_train) 
 
print(b_test) 
 
import numpy as np   
 
import matplotlib.pyplot as plt   
 
import seaborn as seabornInstance  
 
from sklearn.linear_model import LinearRegression 
 
from sklearn import metrics 
 
%matplotlib inline 
 
from sklearn.preprocessing import StandardScaler 
 
sc = StandardScaler() 
 
a_train = sc.fit_transform(a_train) 
 
a_test = sc.transform(a_test) 
 
from sklearn.ensemble import RandomForestRegressor 
 
   regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100) 
 
   regr.fit(a_train, b_train) 
 
   b_pred = regr.predict(a_test) 
 
  
 
   from sklearn.metrics import mean_squared_error as mse 
 
   from sklearn.metrics import mean_absolute_error as mae 
 
   from sklearn.metrics import r2_score 
 
  
 
   print('MSE =', mse(b_pred, b_test)) 
 
   print('MAE =', mae(b_pred, b_test)) 
 
   print('R2 Score =', r2_score(b_pred, b_test)) 
 
from sklearn.svm import SVR 
 
regressorpoly=SVR(kernel='poly',epsilon=1.0) 
 
regressorpoly.fit(a_train,b_train) 
 
pred=regressorpoly.predict(a_test) 
 
print(regressorpoly.score(a_test,b_test)) 
 
print(r2_score(b_test,b_pred)) 
 
!pip install xgboost 
 
from xgboost import XGBRegressor 
 
from sklearn.metrics import mean_absolute_error  
 
XGBModel = XGBRegressor() 
 
XGBModel.fit(a_train,b_train , verbose=False) 
 
  
 
# Get the mean absolute error on the validation data : 
 
XGBpredictions = XGBModel.predict(a_test) 
 
MAE = mean_absolute_error(b_test , XGBpredictions) 
 
print('XGBoost validation MAE = ',MAE) 
 
XGBpredictions 
 
print(r2_score(b_test , XGBpredictions)) 
 
import pickle 
 
# Dump the trained SVM classifier with Pickle 
 
SVM_pkl_filename = 'xgboost_yield_prediction_final.pkl' 
 
# Open the file to save as pkl file 
 
SVM_Model_pkl = open(SVM_pkl_filename, 'wb') 
 
pickle.dump(XGBpredictions, SVM_Model_pkl) 
 
# Close the pickle instances 
 
SVM_Model_pkl.close() 
 
import joblib 
 
  
 
# Save the model as a pickle in a file 
 
joblib.dump(XGBModel, 'xgboost_yield_prediction_final.pkl') 
 
  
 
# Load the model from the file 
 
knn_from_joblib = joblib.load('xgboost_yield_prediction_final.pkl')
