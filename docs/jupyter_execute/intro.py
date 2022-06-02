#!/usr/bin/env python
# coding: utf-8

# 
# # **Pricing Diamonds with the 4 C's** 
# ---
# 
# *By: Kyle W. Brown*

# ## **Problem Statement**
# 
# Pricing diamonds is notoriously difficult, especially for brokers. A diamond can be the same size and same weight but priced thousands of dollars differently. This poses significant challenges to setting the fair market value of diamonds. 
# 

# <center>
# <figure>
# <p float="left">&nbsp;&nbsp;&nbsp;
#   <img src="https://images-aka.jared.com/jared/education/carat-weight-explained/The%20diamond%20carat%20weight%20system%20explained_Img-DiamondWeightChart-Mobile.jpg" width="175" />
#   <img src="https://www.diamondbuild.co.uk/wp-content/uploads/2019/07/Diamond-Clarity-Scale.jpg" width="425" height="300" /> 
# </p>
# </figure>
# </center>

# ## **Value Proposition**
# 
# Give diamond brokers insights into how diamonds are priced. The objective is to provide a tool such as a dashboard that will give greater understanding of how diamonds may be priced. 

# ### **Problems**
# 
# The problems faced during this analysis include:
# 
# 1. Determining the relationship to the 4 C's and pricing, or any identifiable patterns?
# 2. How are the 4 C's distributed across the data?
# 3. How to address the `cut`, `color`, and `clarity` categorical variables?
# 4. How accurate can the price of diamonds be predicted?
# 

# ### **Solutions**
# 
# 1. There appears to be an inverse pricing pattern with the pricing of diamonds with the 4 C's. When comparing the relationship of best `color` with the best `clarity` diamonds, we see that the average `price` (&#36;8,307) significantly higher to the rest of the pivot table. 
# 2. There is correlations among the features, and as a whole the data demostrates not normal distributions. 
# 3. Addressed the `cut`, `color`, and `clarity` categorical variables with ordinal encoding of 1-5 (`cut`), 1-7 (`color`), and 1-8 (`clarity`) from best to worst across the variables. 
# 4. Based on the best performing model, `price` can be predicted quite accurately with a 99% predicted performance.  
# 
# **Suggestable patterns include:**
#   * The inverse pricing pattern is first observed with the average `price` of diamonds by `color` going from lowest to highest, similarities with `cut` and `clarity` continue as well.
#   * The inverse pricing is due the `carat` size increase from best to worst diamonds across `cut`, `color`, and `clarity`.
#   * The worst `cut`, `color`, and `clarity` diamonds have the highest prices.
#   * The best `cut`, `color`, and `clarity` diamonds are among the smallest `carat` in the dataset. 

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()


# In[2]:


diamonds = "https://raw.githubusercontent.com/kyle-w-brown/diamonds-prediction/main/data/diamonds.csv"
df_diamonds = pd.read_csv(diamonds)
df_diamonds.head()


# In[3]:


df_diamonds.shape


# Almost 54,000 rows in the dataset.

# In[4]:


df_diamonds.info()


# Consolidating `x`, `y`, and `z` into `volume`.

# In[5]:


df_diamonds['volume'] = round(df_diamonds['x'] * df_diamonds['y'] * df_diamonds['z'], 2)
df_diamonds.head()


# ## **Data Cleansing**
# 
# ---

# In[6]:


df_diamonds[['x','y','z','volume']] = df_diamonds[['x','y','z','volume']].replace(0, np.NaN)
df_diamonds.isnull().sum()


# Removing missing data

# In[7]:


df_diamonds.dropna(inplace=True)
df_diamonds.isnull().sum()


# ### **Outliers**

# Removing the outliers

# In[8]:


df_diamonds = df_diamonds[(df_diamonds["carat"] <= 5)]
df_diamonds = df_diamonds[(df_diamonds["depth"] < 75) & (df_diamonds["depth"] > 45)]
df_diamonds = df_diamonds[(df_diamonds["table"] < 75) & (df_diamonds["table"] > 45)]
df_diamonds = df_diamonds[(df_diamonds["x"] < 30) & (df_diamonds["x"] > 2)]
df_diamonds = df_diamonds[(df_diamonds["y"] < 30) & (df_diamonds["y"] > 2)]
df_diamonds = df_diamonds[(df_diamonds["z"] < 30) & (df_diamonds["z"] > 2)]
df_diamonds = df_diamonds[(df_diamonds["volume"] < 3500)]
df_diamonds.shape


# ## **Exploration**
# 
# ---

# In[9]:


df = df_diamonds.describe()

heading_properties = [('font-size', '11px')]

cell_properties = [('font-size', '11px')]

dfstyle = [dict(selector="th", props=heading_properties),\
 dict(selector="td", props=cell_properties)]

df.style.set_table_styles(dfstyle)


# Looking at the data we see that the average `carat` size is 0.8 and the largest `carat` is 4.5. The average `price` per diamond is almost &#36;4,000, while the most expensive diamond is priced at &#36;18,823. 

# ### **Exploring the Categorical Variables**

# In[10]:


df_diamonds['cut'].unique()


# In[11]:


df_diamonds['clarity'].unique()


# In[12]:


df_diamonds['color'].unique()


# In[13]:


df_diamonds.describe(include=object)


# **Counting the values per unique feature**

# In[14]:


df_diamonds['cut'].value_counts()


# According to this printout, the total number of diamonds decrease from best (Ideal) to worst (Fair).

# In[15]:


df_diamonds['color'].value_counts()


# In[16]:


df_diamonds['clarity'].value_counts()


# ### **Reordering `cut`, `color`, and `clarity` categorical variables from best to worst**

# In[17]:


df_diamonds['cut'] = pd.Categorical(df_diamonds['cut'], ["Ideal", "Premium", "Very Good", "Good", "Fair"])
df_diamonds = df_diamonds.sort_values('cut')


# In[18]:


df_diamonds['color'] = pd.Categorical(df_diamonds['color'], ["D", "E", "F", "G", "H", "I", "J"])
df_diamonds = df_diamonds.sort_values('color')


# In[19]:


df_diamonds['clarity'] = pd.Categorical(df_diamonds['clarity'], ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])
df_diamonds = df_diamonds.sort_values('clarity')


# **Average price for `cut`, `color`, and `clarity`** 

# In[20]:


cut_avg = round(df_diamonds.groupby('cut')['price'].mean().reset_index(), 2)
cut_avg


# The best `cut` (Ideal) diamonds have the lowest average price. 

# In[21]:


color_avg = round(df_diamonds.groupby('color', as_index=False)['price'].mean(), 2)
color_avg


# The worst `color` (J) diamonds have the highest average price. 

# In[22]:


clarity_avg = round(df_diamonds.groupby('clarity', as_index=False)['price'].mean(), 2)
clarity_avg


# ### **Comparing the 4'C's with Pivot Tables**

# Comparing `cut`, `color`, and `clarity` variables with `price` and `carat` in pivot tables.

# #### **Tables of `Cut` and `Clarity`**

# In[23]:


cut_clarity = df_diamonds.pivot_table('price', index='cut', columns='clarity')

heading_properties = [('font-size', '11px')]

cell_properties = [('font-size', '11px')]

dfstyle = [dict(selector="th", props=heading_properties),\
 dict(selector="td", props=cell_properties)]

cut_clarity.style.set_table_styles(dfstyle)


# We see that the best `cut` (**Ideal**) and the best `color` (**IF**) diamonds are priced at the third lowest across the entire table. 

# In[24]:


cut_clarity_ct = df_diamonds.pivot_table('carat', index='cut', columns='clarity')

heading_properties = [('font-size', '13px')]

cell_properties = [('font-size', '13px')]

dfstyle = [dict(selector="th", props=heading_properties),\
 dict(selector="td", props=cell_properties)]

cut_clarity_ct.style.set_table_styles(dfstyle)


# This table indicates that `carat`'s are increasing from the best `clarity` to the worst `clarity` diamonds. What's interesting is we see this pattern across all `cut` diamonds.  

# #### **Tables of `cut` and `color`**

# In[25]:


cut_color = df_diamonds.pivot_table('price', index='cut', columns='color')

heading_properties = [('font-size', '12px')]

cell_properties = [('font-size', '12px')]

dfstyle = [dict(selector="th", props=heading_properties),\
 dict(selector="td", props=cell_properties)]

cut_color.style.set_table_styles(dfstyle)


# An inverse pricing pattern is emerging with the best `cut` and best `color` diamonds being priced the lowest. As we see there is an increase in price almost among all the features going from best to worst (one would expect to see the opposite).  

# In[26]:


df_diamonds.pivot_table('carat', index='cut', columns='color')


# As we see that with the best `cut` (**Ideal**) and the best `color` (**D**) diamonds have an average price lower than the worst `cut` diamonds. This is due to the best `cut` and `color` diamonds are around a half (0.5) carat, while the worst `cut` (**Fair**) and the worst `color` (**J**) diamonds have the highest average carat at <u>**1.31**</u>.   

# #### **Tables of `color` and `clarity`**

# In[27]:


color_clarity = df_diamonds.pivot_table('price', index='color', columns='clarity')

heading_properties = [('font-size', '11px')]

cell_properties = [('font-size', '11px')]

dfstyle = [dict(selector="th", props=heading_properties),\
 dict(selector="td", props=cell_properties)]

color_clarity.style.set_table_styles(dfstyle)


# The best `color` and best `clarity` diamonds have an average price that is significantly higher than the rest of the variables. Beyond this observation we begin to see that average `price` are among the highest with **SI2** diamonds. 

# In[28]:


color_clarity_ct = df_diamonds.pivot_table('carat', index='color', columns='clarity')

heading_properties = [('font-size', '13px')]

cell_properties = [('font-size', '13px')]

dfstyle = [dict(selector="th", props=heading_properties),\
 dict(selector="td", props=cell_properties)]

cut_clarity_ct.style.set_table_styles(dfstyle)


# The trend continues with the increase of `carat` size from best to worst diamonds. Except when compared with the best `color` and the best `clarity` diamonds, the `carat` size among the best `clarity` (**IF**) diamonds are almost equal to the highest `carat` across the `color` diamonds. In other words, the best `color` (**D**) diamond is only a fraction less than the largest diamond among the best `clarity` (**IF**) category.     

# ## **Visualization**
# 
# ---

# ### **Barplots**

# #### **Barplot of `cut`**

# In[29]:


import plotly.express as px

clr = ['rgb(115, 185, 238)', 'rgb(84, 148, 218)', 'rgb(51, 115, 196)', 
       'rgb(23, 80, 172)', 'rgb(0, 51, 150)', 'rgb(0, 26, 80)']

fig = px.bar(cut_avg, 
             x='cut', 
             y='price', 
             color='cut',
             color_discrete_sequence=clr)

fig.update_layout(showlegend=False)
fig.show()


# #### **Barplot of `color`**

# In[30]:


import plotly.express as px

color = ['rgb(134, 206, 250)', 'rgb(115, 185, 238)', 'rgb(84, 148, 218)', 
         'rgb(51, 115, 196)', 'rgb(23, 80, 172)', 'rgb(0, 51, 150)', 
         'rgb(0, 26, 80)']

fig = px.bar(color_avg, 
             x='color', 
             y='price', 
             color='color',
             color_discrete_sequence=color)

fig.update_layout(showlegend=False)
fig.show()


# #### **Barplot of `clarity`**

# In[31]:


import plotly.express as px

colors = ['rgb(134, 206, 250)', 'rgb(115, 185, 238)', 'rgb(61, 136, 238)', 
          'rgb(84, 148, 218)', 'rgb(51, 115, 196)', 'rgb(23, 80, 172)', 
          'rgb(0, 51, 150)', 'rgb(0, 26, 80)']

fig = px.bar(clarity_avg, 
             x='clarity', 
             y='price', 
             color='clarity',
             color_discrete_sequence=colors)

fig.update_layout(showlegend=False)
fig.show()


# ### **Pairplot**

# In[32]:


ax = sns.pairplot(df_diamonds, 
                  hue= "cut",
                  palette = 'viridis')


# ### **Historgram**

# In[ ]:


df_diamonds.hist(layout=(3,3), figsize=(15,10))
plt.show()


# #### **Histogram `carat`**
# 
# Taking a closer look at `carat`'s distribution.

# In[ ]:


import plotly.express as px

fig = px.histogram(df_diamonds, 
                   x="carat", 
                   marginal="violin",
                   color_discrete_sequence=['rgb(115, 185, 238)'], 
                   hover_data=df_diamonds.columns)
fig.show()


# Notice that the `carat`'s are distributed in increments?

# Normalizing `carat`

# In[ ]:


import plotly.express as px

fig = px.histogram(df_diamonds, 
                   x="carat", 
                   histnorm='probability',
                   color_discrete_sequence=['rgb(115, 185, 238)'] 
                   )
fig.show()


# The highest probability of `carat`'s fall between 0.3 and 1.1

# ### **Boxplots**

# In[77]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

vars = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z', 'volume']
fig = make_subplots(rows=1, cols=len(vars))
for i, var in enumerate(vars):
    fig.add_trace(
        go.Box(y=df_diamonds[var],
        name=var),
        row=1, col=i+1
    )

fig.update_traces(showlegend=False)


# #### **Boxplot of `cut`**

# In[ ]:


import plotly.express as px

fig = px.box(data_frame = df_diamonds,
             x = 'cut',
             y = 'price', 
             color='cut',
             category_orders={"cut": ["Ideal", "Premium", "Very Good", "Good", "Fair"]},
             color_discrete_sequence=clr)

fig.update_layout(showlegend=False)
fig.show()


# #### **Boxplot of `color`**

# In[ ]:


import plotly.express as px

fig = px.box(data_frame = df_diamonds,
             x = 'color',
             y = 'price', 
             color='color',
             category_orders={"color": ["D", "E", "F", "G", "H", "I", "J"]},
             color_discrete_sequence=color)

fig.update_layout(showlegend=False)
fig.show()


# #### **Boxplot of `clarity`**

# In[ ]:


import plotly.express as px

fig = px.box(data_frame = df_diamonds,
             x = 'clarity',
             y = 'price', 
             color='clarity',
             category_orders={"clarity": ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]},
             color_discrete_sequence=colors)

fig.update_layout(showlegend=False)
fig.show()


# It's unique that VS1 and VS2 have the same exact inner quartile ranges considering they may be priced thousands of dollars differently. 

# ### **Ordinal Encoding**
# 
# Creating a rank system for `cut`, `color`, and `clarity`. 

# In[93]:


# Cut rank
cut_two = pd.DataFrame(df_diamonds['cut'])
df_diamonds['cut_rk']= cut_two.replace({'cut':{'Ideal' : 1, 'Premium' : 2, 'Very Good' : 3, 'Good' : 4, 'Fair' : 5}})


# In[94]:


# Color rank
color_two = pd.DataFrame(df_diamonds['color'])
df_diamonds['color_rk']= color_two.replace({'color':{'D' : 1, 'E' : 2, 'F' : 3, 'G' : 4, 'H' : 5, 'I' : 6, 'J' : 7}}) 


# In[95]:


# Clarity rank
clarity_two = pd.DataFrame(df_diamonds['clarity'])
df_diamonds['clarity_rk']= clarity_two.replace({'clarity':{'IF' : 1, 'VVS1' : 2, 'VVS2' : 3, 'VS1' : 4, 'VS2' : 5, 'SI1' : 6, 'SI2' : 7, 'I1' : 8}}) 


# ### **Examining the Ranks of `cut`, `color`, and `clarity`**

# In[ ]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

vars = ['cut_rk', 'color_rk', 'clarity_rk']
fig = make_subplots(rows=1, cols=len(vars))
for i, var in enumerate(vars):
    fig.add_trace(
        go.Box(y=df_diamonds[var],
        name=var),
        row=1, col=i+1
    )

fig.update_layout()
fig.update_traces(showlegend=False)


# ### **Correlation Heatmap**

# In[ ]:


import plotly.express as px
import plotly.graph_objects as go
import numpy as np

df_corr = df_diamonds.corr() 

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x = df_corr.columns,
        y = df_corr.index,
        z = np.array(df_corr),
        colorscale='Viridis'
    )
)


# ## **Models**
# 
# ---

# In[ ]:


import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)


# Slicing the data for numeric columns and removing highly correlated `x`, `y`, and `z`.

# In[96]:


df = df_diamonds.drop(df_diamonds.columns[[1, 2, 3, 7, 8, 9]], axis=1)
df.head()


# With `price` reaching as high as &#36;18,823 and `carat` as low as 0.21, we will need to scale the features.

# In[100]:


from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

df_diamonds_scaled = pd.DataFrame(standard_scaler.fit_transform(df),
                                          columns=df.columns)
df_scaled = df_diamonds_scaled.head()

heading_properties = [('font-size', '13px')]

cell_properties = [('font-size', '13px')]

dfstyle = [dict(selector="th", props=heading_properties),\
 dict(selector="td", props=cell_properties)]

df_scaled.style.set_table_styles(dfstyle)


# ### **Linear Regression**

# In[ ]:


from sklearn.linear_model import LinearRegression
from numpy import *

X = df_diamonds[['carat']]
y = df_diamonds['price']

lr = LinearRegression()   
lr.fit(X, y)             

y_pred = lr.predict(X)

plt.plot(X, y, 'o', color = 'k', label='training data')
plt.plot(X, y_pred, color='#42a5f5ff', label='model prediction')
plt.xlabel('Price per Volume (mass)')
plt.ylabel('Diamond Price ($)')
plt.legend();


# In[ ]:


from sklearn import metrics

# Using scaled features
X = df_diamonds_scaled[['carat']]
y = df_diamonds_scaled['price']

lr = LinearRegression()   
lr.fit(X, y)             

y_pred = lr.predict(X)

print("Mean absolute error (MAE):", metrics.mean_absolute_error(y, y_pred))
print("Mean squared error (MSE):", metrics.mean_squared_error(y, y_pred))
print("Root Mean squared error (RMSE):", np.sqrt(metrics.mean_squared_error(y, y_pred)))
print("R^2:", metrics.r2_score(y, y_pred))


# ### **Multiple Linear Regression**

# In[ ]:


features = ['carat', 'depth','table', 'volume', 'cut_rk', 'color_rk', 'clarity_rk']
X = df_diamonds_scaled[features]
y = df_diamonds_scaled['price']

lr_many_features = LinearRegression()
lr_many_features.fit(X, y);


# In[ ]:


print(('prediction = ' +
       '{} +\n'.format(lr_many_features.intercept_) +
       ' +\n'.join(['{} * {}'.format(n, f) for f, n in zip(features, lr_many_features.coef_)])))


# In[ ]:


print('Multiple features linear model R^2 on training data set: {}'.format(lr_many_features.score(X, y)))


# ### **Random Forest**

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=321)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

forest  = RandomForestRegressor(random_state = random.seed(1234))
model = forest.fit(X_train,y_train)
y_pred = model.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error

print("RMSE: {}".format(np.sqrt(mean_squared_error((y_test),(y_pred)))))
print("R2  : {}".format(np.sqrt(metrics.r2_score((y_test),(y_pred)))))


# In[ ]:


import time

start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


# In[ ]:


plt.rcParams["figure.figsize"] = (8,5.5)

feature_names = [f"feature {i}" for i in range(X.shape[1])]
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


# The feature `volume` appears to the highest importance among the Random Forest model. 

# In[ ]:


from sklearn.inspection import permutation_importance

start_time = time.time()
result = permutation_importance(
    forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)


# In[ ]:


fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()


# The `volume` feature remains the highest with permutation feature importance which indicates do not have a bias toward high-cardinality features and can be computed on a left-out test set. This demonstrates the `volume` overcomes limitations of the impurity-based feature importance. 

# ### **Hyperparameter Tuning Random Forest**

# In[ ]:


n_estimators = [int(x) for x in np.linspace(10,200,10)]
max_depth = [int(x) for x in np.linspace(10,100,10)]
min_samples_split = [2,3,4,5,10]
min_samples_leaf = [1,2,4,10,15,20]
random_grid = {'n_estimators':n_estimators,'max_depth':max_depth,
               'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}

random_grid


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from numpy import *

rf = RandomForestRegressor(random_state = random.seed(1234))
rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               cv = 3)

rf_random.fit(X_train,y_train)
y_pred = rf_random.predict(X_test)

print("RMSE: {}".format(np.sqrt(mean_squared_error((y_test),(y_pred)))))
print("R2  : {}".format(np.sqrt(metrics.r2_score((y_test),(y_pred)))))


# In[ ]:


rf_random.best_params_


# In[ ]:


rf = RandomForestRegressor(max_depth = 30,
                         min_samples_leaf = 2,
                         min_samples_split = 3,
                         n_estimators = 94, 
                         random_state = random.seed(1234))
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

print("RMSE: {}".format(np.sqrt(mean_squared_error((y_test),(y_pred)))))
print("R2  : {}".format(np.sqrt(metrics.r2_score((y_test),(y_pred)))))


# In[ ]:


start_time = time.time()
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


# In[ ]:


feature_names = [f"feature {i}" for i in range(X.shape[1])]
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


# In[ ]:


start_time = time.time()
result = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)


# In[ ]:


fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()


# ### **AutoML using H20**

# In[ ]:


import h2o
h2o.init()


# In[ ]:


diamonds = h2o.import_file("/content/diamonds_new.csv")


# In[ ]:


diamonds.describe()


# In[ ]:


diamonds = diamonds[:, ["carat", "depth", "table", "price", "volume", "cut_rk",	"color_rk", "clarity_rk"]]
print(diamonds)


# #### **GBM Model**

# In[ ]:


from h2o.estimators.gbm import H2OGradientBoostingEstimator

# set the predictor names and the response column name
predictors = ["carat",	"depth",	"table",	"volume", "cut_rk",	"color_rk",	"clarity_rk"]	

response = "price"

# split into train and validation sets
train, valid = diamonds.split_frame(ratios = [.8], seed = 1234)

# train a GBM model
diamonds_gbm = H2OGradientBoostingEstimator(distribution = "poisson", seed = 1234)
diamonds_gbm.train(x = predictors,
               y = response,
               training_frame = train,
               validation_frame = valid)

# retrieve the model performance
perf = diamonds_gbm.model_performance(valid)
perf


# In[ ]:


print('R^2:', diamonds_gbm.r2())
print('R^2 on validation data:', diamonds_gbm.r2(valid=True))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category = matplotlib.cbook.mplDeprecation)


# In[ ]:


diamonds_gbm.varimp_plot();


# #### **AutoML Search**

# In[ ]:


from h2o.automl import H2OAutoML

y = "price"

splits = diamonds.split_frame(ratios = [0.8], seed = 1)
train = splits[0]
test = splits[1]


# In[ ]:


aml = H2OAutoML(max_runtime_secs = 60, seed = 1, project_name = "diamonds_lb_frame")
aml.train(y = y, training_frame = train, leaderboard_frame = test)


# In[ ]:


aml2 = H2OAutoML(max_runtime_secs = 60, seed = 1, project_name = "diamonds_full_data")
aml2.train(y = y, training_frame = diamonds)


# In[ ]:


aml.leaderboard.head()


# In[ ]:


best_model_aml = h2o.get_model(aml.leaderboard[2,'model_id'])

best_model_aml.varimp_plot();


# In[ ]:


print('GBM_2_AutoML_1 R^2:', best_model_aml.r2())
print('GBM_2_AutoML_1 R^2 on validation data:', best_model_aml.r2(valid=True))


# In[ ]:


aml2.leaderboard.head()


# In[ ]:


best_model_aml2 = h2o.get_model(aml2.leaderboard[9,'model_id'])

best_model_aml2.varimp_plot();


# In[ ]:


print('XGBoost_2_AutoML_2 R^2:', best_model_aml2.r2())
print('XGBoost_2_AutoML_2 R^2 on validation data:', best_model_aml2.r2(valid=True))


# In[ ]:


h2o.cluster().shutdown()


# ## **Model Results**
# 
# ---
# 
# 

# |           Model                         |    r2    |
# |:----------------------------------------|:--------:|
# |        Linear_Regression	              |  84.97%	 |   
# |       Multiple_Linear_Regression       	|  90.59%	 |  
# |          XGBoost_2_AutoML_2	            |  98.09%	 |  
# |            GBM_Estimator              	|  98.12%	 |  
# | StackedEnsemble_BestOfFamily_2_AutoML_1	|  98.23%	 |
# |           GBM_2_AutoML_1               	|  98.22%	 |  
# | StackedEnsemble_BestOfFamily_3_AutoML_2	|  98.28%	 |  
# |          Random_Forest	                |  99.07%	 |

# ## **Conclusion**
# 
# ---
# 
# An analysis was performed using the classic `Diamonds` dataset, in which the objective was determining the relationship of 4 C's to price, any identifiable patterns, and how to best price diamonds for brokers. 
# - Through exploration and visualization of the data, observed generalized pattern of inverse pricing accompanied with not normal distributions. 
# - The clearest indication is the combination of best `color` and best `clarity` diamonds are priced significantly higher, while `cut`, `color`, `clarity` are priced highest from the worst diamonds.   
# - Of the 4 C's `carat`'s coefficient level in the multiple linear regression, and among the variable importance compared favorable against the other 4 C's.
# - After scaling the features, the baseline linear regression captured a modest 84.97% accuracy, while the Random Forest model scored the highest with 99.07%.
# - The final deliverable was a Tableau Dashboard, to assist brokers with visualizations and a potential pricing mechanism to the `Diamonds` dataset. The dashboard can be viewed [here](https://public.tableau.com/app/profile/kyle.w.brown/viz/Diamonds_16516232030110/DiamondsOverview?publish=yes/) 
