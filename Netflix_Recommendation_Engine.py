#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Libraries Needed#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


# In[2]:


#Importing data set we have imported Netflix Prize data from kaggle#
#There are 4 data sets in Netflix Prize Dataset we have uploaded only one all four can be uploaded and a better recommendation can be made#
#Only one data set is uploaded due to comp power limitations#  
netflix_dataset = pd.read_csv('combined_data_1.txt', header = None, names = ['Cust_ID','Rating',], usecols = [0,1,])


# In[3]:


netflix_dataset.head()  #Printing 1st 5 readings#


# In[4]:


netflix_dataset.dtypes #Data type for values of netflix_dataset#
#Object means String data type#


# In[5]:


#If the Rating value is shown as object(String) then the following instruction will convert it to float data type#
#Then grouped into the Rating column all in one#
netflix_dataset['Rating'] = netflix_dataset['Rating'].astype(float) 


# In[6]:


netflix_dataset.dtypes #Its data type#


# In[7]:


#To check the number of rows and columns of data we use the following instruction#
netflix_dataset.shape 


# In[8]:



#We are going to separate the ratings and customer id data in the netflix_daatset#
stars = netflix_dataset.groupby('Rating')['Rating'].agg(['count']) 
#To Find no of people who have given different ratings of 1, 2, 3, 4, 5#


# In[9]:


stars


# In[10]:


#to calculate no of movies in dataset#
movie_count = netflix_dataset.isnull().sum() 
#In netflix_dataset we see that movie id is associated with NaN in the ratings columns which is why we are checking for no of NaN to determine the no of movies#
movie_count


# In[11]:


#To get the no of movies from movie_count and eliminating the Cust_ID value we use#
movie_count = netflix_dataset.isnull().sum()[1]#WE are using only the 1st index value#
movie_count


# In[12]:


#To determine the no of coustomers we use the netflix_dataset and we determine the no of customers who have given a rating #
#These are the ones who do not have an NaN value in the ratings columns#
#In case of repeation of ratings meanings same id multiple times is added to the no of customers#
#Which is why we use the nunique() func and remove repeation of Customer_ID#
coustomer_count = netflix_dataset['Cust_ID'].nunique()
coustomer_count
#The above value also contains movie_id which has to be eliminated#


# In[13]:


#By removing the movie_count from above customer_count we get the proper customer_count#
coustomer_count = netflix_dataset['Cust_ID'].nunique() - movie_count
coustomer_count


# In[14]:


#To get the total no of ratings given by all coustomer for all the movies#
rating_count = netflix_dataset['Cust_ID'].count() - movie_count
#movie_count is subtracted to get proper rating_count#
rating_count


# In[15]:


#Ploting a horizontal graph of generated data. Vertical bars are too big to plot#
ax = stars.plot(kind = 'barh', legend =False, figsize =(15,10)) #basrh->horizontal bar graph
plt.title(f'Total pool: {movie_count}->Movies, {coustomer_count}->Coustomers, {rating_count}->Ratings Given')
plt.grid(True)


# In[16]:


#We are separating the movie_id column from rating column of netflix_dataset#
#Add another column that will have only movie id#
#We are basically going to find out the position of the movie_id in the netflix_dataset#
#We are calculating no of null values in ratings column in netflix_dataset#
df_nan = pd.DataFrame(pd.isnull(netflix_dataset.Rating))


# In[17]:


df_nan.head()


# In[18]:


#Extracting only the TRUE values from the df_nan#
df_nan = df_nan[df_nan['Rating']==True]


# In[19]:


#To find no of rows in df_nan#
df_nan.shape


# In[20]:


df_nan.head() #The positions of Movie ID in netflix_dataset stored in df_nan#
#0-547->Movie_Id_1, 548-693->Movie_ID_2, ........#


# In[21]:


df_nan = df_nan.reset_index()#By using reset func we are getting the index values in a column#
df_nan.head()


# In[22]:


#Creating a numpy array that will contain rating of movie 1 from o to 547, for 2 from 548 to 694 and so on#
movie_np=[]
movie_id=1
for i,j in zip(df_nan['index'][1:], df_nan['index'][:-1]): #zip func helps convert data in df_nan into a tuple#
   temp = np.full((1,i-j-1),movie_id)
   movie_np = np.append(movie_np, temp)
   movie_id+=1

#Account for last record and corresponding length
#numpy approach
last_record =  np.full((1,len(netflix_dataset) - df_nan.iloc[-1,0]-1),movie_id)
movie_np = np.append(movie_np, last_record)
print(f'movie numpy:{movie_np}')
print(f'Length: {len(movie_np)}')


# In[23]:


x = zip(df_nan['index'][1:], df_nan['index'][:-1])#Ex of working of a tuple#


# In[24]:


tuple(x) #Making it a tuple#
#Show that from o to 547 is movie_id1, from 548 to 694 is movie_id2 and so on till end of the df_nan#


# In[25]:


temp = np.full((1,547),1) #working of full func#
temp #We are initailizing all the values as 1 from 1-547#


# In[26]:


netflix_dataset = netflix_dataset[pd.notnull(netflix_dataset['Rating'])]
netflix_dataset['Movie_Id'] = movie_np.astype(int)
netflix_dataset['Cust_ID'] = netflix_dataset['Cust_ID'].astype(int)
print("Now the dataset wioll look like: ")
netflix_dataset.head()


# In[27]:


#We will remove all user ids that have rated less than 3 movies cause these ratings do not present us with enough feedback#
#We will also eliminate those movies which have been rated only by few users#
f=['count','mean']


# In[28]:


#Grouping of the no of ratings given to each movie and the mean of all ratings#
dataset_movie_summary = netflix_dataset.groupby('Movie_Id')['Rating'].agg(f)
dataset_movie_summary


# In[29]:


#Storing all Movie_Id indexes in data_movie_summary and all the index values are converted to int and stored#
dataset_movie_summary.index = dataset_movie_summary.index.map(int)


# In[30]:


#Creating a benchmark basically by these statements we are elliminating the poor reviews posted and are considering
#only the positive reviews
movie_benchmark = round(dataset_movie_summary['count'].quantile(0.7),0) #Do not take 0.5 it will divide the table into 2 parts#
#The data is divided into a 70:30 ratio by the quantile func#
movie_benchmark#By taking 0.7 we are dividing the table by 70% in  the count column#


# In[31]:


dataset_movie_summary['count']


# In[32]:


#Removing movies with less rating#
drop_movie_list = dataset_movie_summary[dataset_movie_summary['count'] < movie_benchmark].index
drop_movie_list


# In[33]:


dataset_cust_summary = netflix_dataset.groupby('Cust_ID')['Rating'].agg(f)
dataset_cust_summary


# In[34]:


dataset_cust_summary.index = dataset_cust_summary.index.map(int)


# In[35]:


cust_benchmark = round(dataset_cust_summary['count'].quantile(0.7),0)
cust_benchmark


# In[36]:


#Removing inactive users#
drop_cust_list = dataset_cust_summary[dataset_cust_summary['count']>cust_benchmark].index
drop_cust_list


# In[37]:


#We will remove all movie and customer below benchmark
print('The Original dataframe has: ',netflix_dataset.shape,'shape')


# In[38]:


# ~ represents elimination, the movies with low rating and inactive users are removed from netflix_dataset#
netflix_dataset = netflix_dataset[~netflix_dataset['Movie_Id'].isin(drop_movie_list)]
netflix_dataset = netflix_dataset[~netflix_dataset['Cust_ID'].isin(drop_cust_list)]
print('After triming the shape is : {}',format(netflix_dataset.shape))


# In[39]:


netflix_dataset.head()


# In[40]:


#We now prepare the dataset for SVD(Singular Value Decomposition Algoritham) and it takes matrix as input#
#so for ip we will convert dataset into sparse matrix#
df_p = pd.pivot_table(netflix_dataset, values='Rating', index='Cust_ID', columns='Movie_Id')
#pivot_table func helps convert the data structure into a sparse matrix#
print(df_p.shape)


# In[41]:


df_p.head()


# In[42]:


#Importing movie_titles file#
df_title = pd.read_csv('movie_titles.csv', encoding='ISO-8859-1', header=None, names=['Movie_Id', 'Year','Name'])
#We use encoding because in movie_titles file movies name can include certain symbols like $, & which is why we use the encoding to recognise these symbols#
#header is equal to none to ignore the 1st column reading of movie_titles#
df_title.set_index('Movie_Id', inplace=True)


# In[43]:


df_title.head(20)


# In[44]:


#Helps insatll the surprise package#
#pip install scikit-surprise


# In[45]:


#Libraries needed to build model#
import math
import re
from scipy.sparse import csr_matrix
import seaborn as sns
from surprise import Reader, Dataset, SVD
# SVD package helps implement svd algorithm#
# Dataset package helps read the above data because svd cannot read it in its present form# 
from surprise.model_selection import cross_validate
# cross_validate helps determine the crossvalidation of dataset#


# In[46]:


#helps read the dataset for svd algorithm# 
reader = Reader()


# In[47]:


#ensures that model is built only on 1st 100 data rows for quiker run time#
data = Dataset.load_from_df(netflix_dataset[['Cust_ID','Movie_Id','Rating']][:100], reader)


# In[48]:


#Creating svd algorithm#
svd=SVD()
cross_validate(svd, data, measures=['RMSE','MAE'], cv=3, verbose=True) 
#csv can vary from 5-10#
#while training the error obtained by the svd algo is displayed by RMSE ->Root Mean square Error and MAE ->Mean Absolute Error#
#first the error is calculated then the square root of error is calculated and stored in root mean square error#
#cross_validate uses K-Fold Algo and divides the given data set into 3 folds#
#for 1st fold 1,2,3#
#verbose = 'True" helps print the o/p #


# In[49]:


netflix_dataset.head()


# In[50]:


#We use user_712664 and try to recommend some movies based on past data#
dataset_712664 = netflix_dataset[(netflix_dataset['Cust_ID']==712664)&(netflix_dataset['Rating']==5)]
dataset_712664 = dataset_712664.set_index('Movie_Id')
dataset_712664 = dataset_712664.join(df_title)['Name']
dataset_712664


# In[51]:


user_712664 = df_title.copy()
user_712664


# In[52]:


#we are going to set the movie_id as a column#
user_712664 = user_712664.reset_index()
user_712664


# In[53]:


#Removing all the movies listed in the drop_movie_list from the recommended list#
user_712664 = user_712664[~user_712664['Movie_Id'].isin(drop_movie_list)]
user_712664


# In[54]:


#We will now train our algorithm with the whole dataset#
data = Dataset.load_from_df(netflix_dataset[['Cust_ID','Movie_Id','Rating']], reader)


# In[55]:


#building the trainset using surprise package#
trainset = data.build_full_trainset()#this will modify our data into a training set to be passed into svd model#
svd.fit(trainset)#this trains our model on top of our dataset#


# In[56]:


#Now we will try to predict our model#
user_712664['Estimate_Score'] = user_712664['Movie_Id'].apply(lambda x: svd.predict(712664, x).est)
#lambda is an anonyamous fuc#
#we use est func to get recomendation score#
user_712664 = user_712664.drop('Movie_Id',axis=1)
#We are now droping the Movie_Id column from the dataset#


# In[57]:


user_712664 = user_712664.sort_values('Estimate_Score')
print(user_712664.head(10))


# In[58]:


user_712664 = user_712664.sort_values('Estimate_Score', ascending=False)
print(user_712664.head(10))


# In[ ]:




