#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load data
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


train_df = pd.read_csv('./train_data/3.csv')
test_df = pd.read_csv('./test_data/3.csv')

# Training set and test set size
print(train_df.shape,test_df.shape)


# In[2]:


# View the first row of data
train_df.head(1)


# In[3]:


test_df.head(1)


# In[4]:


# Does it contain missing values
train_df.isna().sum()


# In[5]:


test_df.isna().sum()


# In[6]:


# Data details
train_df.info()


# In[7]:


test_df.info()


# In[8]:


train_df.columns


# In[9]:


train_label = train_df[['label']]
train_data = train_df.drop(['label','position'],axis=1)

test_label = test_df[['label']]
test_data = test_df.drop(['label','position'],axis=1)


# In[10]:


# Polyline visualization of train_label and test_label
plt.plot(train_label, label='Train Label')
plt.plot(test_label, label='Test Label')
plt.legend()
plt.title('Train and Test Label Visualization')
plt.show()


# XGBoost is a gradient boosting tree method built on decision trees. Gradient boosted trees are an ensemble learning method that improves the predictive performance of a model by incrementally building multiple decision trees. Each tree attempts to correct the prediction error of the previous tree.


# In[11]:


# Training using XGBoost
model = xgb.XGBRegressor()
model.fit(train_data, train_label)

# res = pd.DataFrame()
res = pd.read_csv('./res/predict.csv')
# predict
predictions = model.predict(test_data)
res['3_predict'] = predictions
res.to_csv('./res/predict.csv',index = False)
# Calculate MSE, MAE and R^2
mse = mean_squared_error(test_label, predictions)
mae = mean_absolute_error(test_label, predictions)
r2 = r2_score(test_label, predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R^2): {r2}")


# MSE (Mean Squared Error) is an indicator that measures the mean squared difference between the predicted value and the actual value. It calculates the difference between the model's predicted and actual values and averages the squares of these differences. The smaller the MSE value, the better, indicating that the model's predictions fit better with the actual values.
# MAE (Mean Absolute Error) is an indicator that measures the average absolute difference between the predicted value and the actual value. It calculates the average of the absolute differences between the model's predicted and actual values. The smaller the MAE value, the better, indicating that the model's predictions fit better with the actual values.
# The coefficient of determination (R2) is an index that measures the fitness of the model. It indicates the proportion of the variance of the model's explanatory variables in the total variance. The value range of R-squared is usually between 0 and 1. The closer to 1, the better the model fits the data.  
# Taken together, the MSE and MAE values are relatively small, and the R-squared value is close to 1, which indicates that the model performs well on this data set and can fit the target variable well.

# In[12]:


# Feature importance visualization
xgb.plot_importance(model)
plt.savefig('./res/3_feature_importance.png',dpi = 300)
plt.show()


# Visualizing feature importance is a method for understanding how much each feature (input variable) in a machine learning model contributes to the model's predictions. This visualization provides information about which features have the greatest impact on the model's performance, aiding in feature selection and explaining model behavior. In a bar chart, each feature is represented as a column on the horizontal axis, and the height of the column represents the importance score of the feature. Typically, features with higher importance scores have higher bars.

# In[13]:


# Perform polyline visualization on the predicted label and test_label
plt.plot(test_label, label='Test Label')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.title('Test Label and Predictions Visualization')
plt.savefig('./res/3_Predictions_Visualization.png',dpi = 300)
plt.show()


# In[ ]:




