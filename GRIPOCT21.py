GRIP OCT21' - The Spark Foundation
Author:- Atharva More
Domain:- DATA SCIENCE AND BUSINESS ANALYTICS
Task 1 : PREDICTION USING SUPERVISED LEARNING
Language : Python
Dataset Link : http://bit.ly/w-data

# In[17]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split as tts


# In[18]:


# reading the data
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
data.head(10) 


# In[19]:


data.info()


# In[20]:


#to check whether any duplicate value or missing value is present or not
data.isnull().sum()


# In[21]:


#analysis on data
data.describe()


# # Plotting the dataset

# In[22]:


data.plot(x='Hours',y="Scores", style='go')
plt.title("Prediction")
plt.xlabel("Hours_Studied")
plt.ylabel("Test_score")
plt.show()


# In[23]:


sns.boxplot(data=data[['Hours','Scores']])


# # Preparing Data
# 

# In[24]:


x = data.iloc[:,:-1].values
y= data.iloc[:,1].values


# In[25]:


x


# In[26]:


y


# # Implementing Training Sets and Test Sets

# In[27]:


#we are going to use 20% of our data for testing and rest for training the dataset
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.20,random_state = 0)


# # Training the Algorithm

# In[28]:


#predicting the percantage of the marks 
reg = lr()
reg.fit(x_train,y_train)


# In[29]:


l = reg.coef_*x + reg.intercept_
data.plot.scatter(x="Hours",y="Scores")
plt.plot(x,l)
plt.grid()
plt.show()


# In[30]:


y_pred=reg.predict(x_test)
print(y_pred)


# In[31]:


#final prediction for the case that if a student studies 9.25 hrs/day

h = np.array([[9.25]])
p = reg.predict(h)
print("No of hours = ", h[0][0])
print("Predicted Score =",p[0])


# # Evaluating the Model 

# In[32]:


print("Mean Absolute Error = ", metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:





# In[ ]:





