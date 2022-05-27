#!/usr/bin/env python
# coding: utf-8

#  ## Project - 1

# In[253]:


import pandas as pd
import numpy as np
from sklearn import datasets


# In[254]:


housing = pd.read_csv("data.csv")


# In[255]:


print(housing.shape)


# In[256]:


housing.describe()


# In[257]:


import matplotlib.pyplot as plt


# In[258]:


housing.hist(bins=50, figsize=(20, 15))


# split_train_using numpy by creating a fucntion.

# In[259]:


# import numpy as np
# def split_train_test(data, test_ratio):
#     shuffled = np.random.permutation(len(data))
#     np.random.seed(42)
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size:]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]


# In[260]:


from sklearn.model_selection import train_test_split
train_set,test_set= train_test_split(housing, test_size =0.2)


# In[261]:


print(f"rows in train set : {len(train_set)}\nRows in test set : {len(test_set)}")


# ## Train-test spiliting

# In[262]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"rows in train set : {len(train_set)}\nRows in test set : {len(test_set)}")


# stratified spilit for equal spiliting of 'CHAS'

# In[263]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
   strat_train_set = housing.loc[train_index]
   strat_test_set = housing.loc[test_index]


# In[264]:


strat_train_set['CHAS'].value_counts()


# In[265]:


# 95/7


# In[266]:


# 376/28


# In[267]:


housing = strat_train_set.copy()


# LOOKING FOR CORRELATION'S

# In[268]:


# correlations can help us examine data , watchout for outliers , for socanning and for relations btwn other features


# In[269]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
# 1 is strong positive correlation
# rm is strongly positive which is rooms per dwelling which means it is directly propotional with MEDV(price)


# In[270]:


from pandas.plotting import scatter_matrix


# In[271]:


attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(housing[attributes], figsize=(12,8))


# <!-- BY scanning all the graphs we can clearly see that RM and MEDV is showing very good result and with that LSTAT and MEDV is also showing very good result , lets analyse the RM and MEDV more for clear vision -->

# In[272]:


housing.plot(kind ="scatter", x = "RM", y = "MEDV")
housing.plot(kind ="scatter", x = "RM", y = "TAX")


#  <!-- after analysing the RM - MEDV we can observe some outliers(at 50) in our data sets so make our model more accurate we can remove those outliers from dataset so the model give more accurate calcluations... -->  

# Trying our attribute combinations 

# In[273]:


housing['TAXRM'] = housing['TAX']/housing['RM']
housing.plot(kind = "scatter", x = "TAXRM", y = "MEDV")


# In[274]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()
# housing.describe()
# housing_labels


# creating a automated pipeline for future datasets where values are missing (for example in RM)

# In[275]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[276]:


imputer.statistics_.shape


# In[277]:


x = imputer.transform(housing)
housing_tr = pd.DataFrame(x, columns = housing.columns)


# In[278]:


housing_tr.describe()


# In[279]:


housing_tr.head(10)


# -- sckit learn dedign reference https://medium.com/analytics-vidhya/scikit-learn-design-with-easy-explanation-b3bcb060580

# ## **creating pipeline

# In[280]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[281]:


my_pipeline = Pipeline([
   ('imputer', SimpleImputer(strategy='median')),
   ('std_scaler',StandardScaler()),
])


# In[282]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[283]:


housing_num_tr.shape


# MODEL SELECTION FOR DATASET

# In[284]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[285]:


some_data = housing.iloc[:5]


# In[286]:


some_labels = housing_labels[:5]


# In[287]:


prepared_data = my_pipeline.transform(some_data)


# In[288]:


model.predict(prepared_data)


# In[289]:


list(some_labels)


# evaluating model

# In[290]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
print(f"mse is : {rmse}")


# cross valuation

# In[301]:


from sklearn.model_selection import cross_val_score
# from sklearn.metrics import get_scorer_names
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring='neg_mean_squared_error', cv=10)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse_scores = np.sqrt(-scores)


# In[292]:


def print_scores(scores):
   print("scores is : ", scores)
   print("mean is : ", scores.mean())
   print("standard deviation is : ", scores.std())


# In[293]:


print_scores(rmse_scores)


# ## saving the model

# In[294]:


from joblib import dump, load
dump(model, "finalmodel.joblib")


# In[295]:


prepared_data[0]


# ## testing the model 

# In[299]:


X_test = strat_test_set.drop("MEDV", axis = 1)
y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[313]:


def test_result(final_mse,final_rmse,final_predictions):
   print(f"final mse is : {final_mse} \n")
   print(f"final rmse is : {final_rmse} \n")
   print(f"final predictions is : {final_predictions} \n")


# In[314]:


test_result(final_mse,final_rmse,final_predictions)


# In[ ]:




