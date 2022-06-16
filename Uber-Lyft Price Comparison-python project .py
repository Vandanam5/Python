#!/usr/bin/env python
# coding: utf-8

# # Uber Lyft Price Prediction 

# ![image.png](attachment:image.png)

# Both Uber and Lyft are ride-hailing services that allow users to hire vehicles with drivers through websites or mobile apps. Uber is a global company available in 69 countries and around 900 cities worldwide. Lyft, on the other hand, operates in about 644 cities in the US and 12 cities in Canada only. Yet, in the US, it’s the second-largest ridesharing company with a 31% market share. From booking the cab to paying the bill, both services have almost similar core features. But there are some unique cases where the two ride-hailing services come neck to neck. One such is pricing, especially dynamic pricing called “surge” in Uber and “Prime Time” in Lyft. We have an interesting dataset with data from Boston (US), which we will analyze to understand the factors affecting the dynamic pricing and the difference between Uber and Lyft’s special prices.

# # Data Set

# The datasets used in this article have been imported from: [Kaggle] The data has been collected from different sources, including real-time data collection using Uber and Lyft API (Application Programming Interface) queries. The dataset covers Boston’s selected locations and covers approximately a week’s data from November 2018. After loading the dataset, we found two files available. The first is the weather.csv, and the second is cab_rides.csv.

# # Importing Libraries 

# In[1]:



import warnings
warnings.filterwarnings('ignore')

# Importing the numpy and pandas package
import pandas as pd
import numpy as np
# Importing Matplotlib and Seaborn for Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Data with CSV format

# In[2]:


# loading csv data to dataframe 
#This project consist of 2 data sets cab_rides and weather
#Importing cab rides dataset
cab=pd.read_csv(r"C:\Users\vanda\OneDrive\Desktop\Python project\Data\cab_rides.csv")
#Importing weather dataset
weather=pd.read_csv(r"C:\Users\vanda\OneDrive\Desktop\Python project\Data\weather.csv")


# # EDA(Exploratory Data Analysis)

# # Getting Familiar With The Date Sets

# # Finding shape of cab and weather dataset

# In[3]:


#Finding shape of cab and weather dataset
cab.shape,weather.shape


# # Finding Head oF cab Dataset

# In[4]:


#Finding Head oF cab Dataset
cab.head()


# # Finding tail of cab Dataset

# In[5]:


#Finding tail oF cab Dataset
cab.tail()


# In[6]:


cab


# # Finding Head of weather Dataset

# In[7]:


#Finding Head of weather Dataset
weather.head()


# In[8]:


weather


# # Finding information about the variables available in cab  and weather dataset

# In[9]:


cab.info()


# In[10]:


cab.isna().sum()


# In[11]:


cab=cab.dropna(axis=0).reset_index(drop=True)#axix zero as we are dropping rows


# In[12]:


cab.isna().sum()


# In[13]:


weather.info()


# In[14]:


weather.isna().sum()


# In[15]:


weather=weather.fillna(0)


# In[16]:


weather.isna().sum()


# In[17]:


weather.groupby('location').mean()


# In[18]:


avg_weather=weather.groupby('location').mean().reset_index(drop=False)
avg_weather=avg_weather.drop('time_stamp',axis=1)


# In[19]:


avg_weather


# In[22]:


#Merging Data Frames 
#Renaming column location to source 
avg_weather.rename(columns={"location":"source"},inplace=True)


# In[23]:


avg_weather


# In[24]:


df=cab.merge(avg_weather,on="source")


# In[25]:


df


# In[70]:





# In[26]:


df['time_stamp']=pd.to_datetime(df['time_stamp'],unit='ms')
df.head()


# In[27]:


df.dtypes


# In[28]:


df['Years'] =  df.time_stamp.dt.year
df['Month'] =  df.time_stamp.dt.month_name()
df['Day'] =  df.time_stamp.dt.day
df['Hour'] =  df.time_stamp.dt.hour
df['Minute'] =  df.time_stamp.dt.minute
df['Second'] =  df.time_stamp.dt.second
df['Weekday'] =  df.time_stamp.dt.day_name()


# In[70]:


df


# In[71]:


df.head()


# In[29]:


df=df.drop(['id','product_id'],axis=1)


# In[73]:


df


# In[30]:


df.info()


# In[75]:


#Checking for Duplicates
df.duplicated().sum()


# In[76]:


df.drop_duplicates(keep=False,inplace=True)


# In[77]:


df.duplicated().sum()


# # Finding count (number of non_missing values),unique values(or levels), top(mode) and freq(fequency of mode)

# In[78]:


#finding count (number of non_missing values),unique values(or levels), 
#top(mode) and freq(fequency of mode)
#finding unique values
df.astype('object').describe().transpose()#transpose to swap rows and columns


# # Getting the summary of Data

# In[79]:


# Getting the summary of Data
pd.options.display.float_format = "{:.2f}".format
df.describe().transpose()# for numeric columns


# # Finding unique values

# In[80]:


#Finding unique values
df.apply(lambda x: len(x.unique()))


# # Find frequency of each categorical column 

# In[81]:


#Find frequency of each categorical column 

df_cat=df.select_dtypes(include='object')
for i in df_cat.columns:
    print(i,":")
    print(df[i].value_counts())
    print("--------------------------")


# # Visualization and analysis

# In[31]:


cab_type=df['cab_type'].value_counts()
cab_type


# In[33]:


#Pie chart
plt.figure(figsize=(8,8))
plt.pie(x=df['cab_type'].value_counts(),labels=df['cab_type'].value_counts().index, autopct='%0.2f%%')
plt.title("Distribution of Cab Types")
plt.show()


# # Distribution of distance 

# In[34]:


sns.displot(data=df, x='distance',hue='cab_type', bins=60, color='pink', height=8)


# # separating the lyft and uber datas for further analysis

# In[77]:


#separating the lyft and uber datas
lyft  = df[df['cab_type']== 'Lyft'] 
uber  = df[df['cab_type']== 'Uber']


# In[82]:


uber


# In[83]:


lyft


# # What is the distribution of number of rides by each category in uber and lyft?

# In[37]:


#Different car models available 
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))
sns.countplot(lyft['name'], ax=ax1)
ax1.set_title('Distribution of Rides By each Car Category in Lyft', fontsize=20)
sns.countplot(uber['name'], ax=ax2)
ax2.set_title('Distribution of Rides By each Car Category in Uber', fontsize=20)
ax1.set(xlabel='Lyft Car Types', ylabel='Total')
ax2.set(xlabel='Uber Car Types', ylabel='Total')


# The dataset has almost equal distribution of cab types for both uber and lyft company. Uber has around 5000 to 6000 more rides than Lyft for each category.

# # What is the price distribution for each cab category in Uber and Lyft?

# In[55]:


#price distribution for uber 
x= df. groupby('cab_type'). sum()
x.iloc[:, [1]]


# In[38]:


#price distribution for uber 
x=uber. groupby('name'). sum()
x.iloc[:, [1]]


# In[40]:


#price distribution for Lyft 
x=lyft. groupby('name'). sum()
x.iloc[:, [1]]


# In[47]:


#pie chart percentage distribution for price in Uber cab category 
fig = px.pie(uber, values="price", names="name",width=800, height=400)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# In[48]:


#pie chart percentage distribution for price in lyft cab category 
fig = px.pie(lyft, values="price", names="name",width=800, height=400)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# Sharing cab ride categories that is 'shared' for lyft and 'Uberpool' in Uber offers cheaper price for customers compared to other categories. Out of which lyft has cheaper price compared to uber in shared ride category.And in both category luxury car rides 'Lux Black XL' and 'Black SUV'is most expensive. In other categories both uber and lyft seems to share similar price range. Further analysis price is required based on distance covered to get a clear idea of which cab type is cheaper.

# In[35]:


#Average price range of cab rides
Avg_Price=df['price'].mean()
Avg_Price


# In[36]:


#Minimum price range of cab rides
Min_Price=df['price'].min()
Min_Price


# In[37]:


#Maximum price range of cab rides
Max_Price=df['price'].max()
Max_Price


# In[50]:


#Average, minimum and maximum price for each cab Type
Average_Price =df.groupby(by='cab_type')['price'].mean()
Minimum_Price = df.groupby(by='cab_type')['price'].min()
Maximum_Price = df.groupby(by='cab_type')['price'].max()
print(Average_Price)
print(Minimum_Price)
print(Maximum_Price)


# In[51]:


#Average, Minimum and Maximum price for each cab Type in uber 
Average_Price =uber.groupby(by='name')['price'].mean()
Minimum_Price = uber.groupby(by='name')['price'].min()
Maximum_Price = uber.groupby(by='name')['price'].max()
print(Average_Price)
print(Minimum_Price)
print(Maximum_Price)


# In[52]:


#Average, Minimum and Maximum price for each cab Type in lyft 
Average_Price =lyft.groupby(by='name')['price'].mean()
Minimum_Price =lyft.groupby(by='name')['price'].min()
Maximum_Price =lyft.groupby(by='name')['price'].max()
print(Average_Price)
print(Minimum_Price)
print(Maximum_Price)


# # What is the distribution of distance for uber and lyft cabs ?

# In[53]:


total_distance_data = df['distance'].describe() 
lyft_distance = lyft['distance'].describe()
uber_distance = uber['distance'].describe()
stats = pd.DataFrame({'Total': total_distance_data.values,
                  'Lyft': lyft_distance.values,
                  'Uber': uber_distance.values}, index= ['Count', 'Mean', 'Std. Dev.', 'Min', '25%', '50%', '75%', 'Max'])
stats


# In[56]:


#What is the distribution of distance for uber and lyft cabs ?
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))
sns.boxplot(lyft['distance'], ax=ax1,palette="Set3")
ax1.set_title('Lyft')
ax1.set_xlim(0, 7)
sns.boxplot(uber['distance'], ax=ax2,palette="Set1")
ax2.set_xlim(0, 7)
ax2.set_title('Uber')


# # How is the cab price affected as the distance travelled increases ?
# 

# Doing segmentation of distance travelled to categories of short distance,medium distance and long distance and mapping iit to the dataframe to compare price and distance 

# In[60]:


def fdist(x):
    if (x>=0.01) & (x<3):
        distance_segment = 'Short Distance'
    elif (x>=3) & (x<6):
        distance_segment = 'Medium Distance'  
    else:
         distance_segment = 'Long Distance'
    return  distance_segment


# In[61]:


#mapping it to dataframe
df['distance_segment'] = df.distance.map(fdist) 


# In[66]:


df.head(50000)


# # What is the Distribution of Price Per Distance Travelled for each cab type as well as their categories ?

# In[57]:


#Distribution of Price per Distance travelled for uber and lyft
plt.figure(figsize=(16, 6))
sns.set_theme( palette="pastel")
sns.barplot(x="distance_segment", y="price",
            hue="cab_type", 
            data=df,ci=None)
sns.despine(offset=10, trim=True)
plt.title('Distribution of Price per Distance travelled')


# In[58]:


#Distribution of Price per Distance travelled for cab type subcategories 
plt.figure(figsize=(16, 6))
sns.set_theme( palette="pastel")
sns.barplot(x="distance_segment", y="price",
            hue="name", 
            data=df,ci=None)
sns.despine(offset=15, trim=True)
plt.title('Distribution of Price per Distance travelled')


# # How Price is affected during different time period in a day?

# In[61]:


ticks = [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
labels = ["12AM", "1AM", "2AM", "3AM", "4AM","5AM","6AM","7AM","8AM","9AM","10AM","11AM","12PM","1PM","2PM","3PM","4PM","5PM","6PM","7PM","8PM","9PM","10PM","11PM"]

plt.figure(figsize=(16, 6))
sns.lineplot(x="Hour", y="price", data=uber,color="blue")
plt.xticks(ticks,labels)


# In[133]:


ticks = [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
labels = ["12AM", "1AM", "2AM", "3AM", "4AM","5AM","6AM","7AM","8AM","9AM","10AM","11AM","12PM","1PM","2PM","3PM","4PM","5PM","6PM","7PM","8PM","9PM","10PM","11PM"]

plt.figure(figsize=(16, 6))
sns.lineplot(x="Hour", y="price", data=lyft,color="red")
plt.xticks(ticks,labels)


# In[73]:


def ftime(x):
    if (x>=4) & (x<12):
        time_segment = 'Morning'
    elif (x>=12) & (x<16):
         time_segment = 'Afternoon'
    elif (x>=16) & (x<21):
         time_segment = 'Evening'
    elif (x>=21) & (x<24):
         time_segment = 'Night'     
    else:
         time_segment = 'Late Night'
    return  time_segment 


# In[74]:


#mapping it to dataframe
df['time_segment'] = df.Hour.map(ftime)


# In[76]:


df


# In[64]:


dk  = df[df['price']<= 50] 


# # Average, Minimum and Maximum Price during different time segments in a day for uber 

# In[79]:


#Distribution of cabs during different time segments through out the day
g1=sns.catplot(x='time_segment',data=uber,kind='count')
g1.set_axis_labels('Uber', 'Total')
g1.set(title='Uber cabs taken during different time segments')
sns.set(font_scale = 0.75)
g2 = sns.catplot(x='time_segment',data=lyft,kind='count')
g2.set_axis_labels('Lyft', 'Total')
g2.set(title='Lyft Cabs taken during different time segments ')
plt.show(g1,g2)


# In[83]:


Average_Price_uber =uber.groupby(by='time_segment')['price'].mean()
Minimum_Price_uber =uber.groupby(by='time_segment')['price'].min()
Maximum_Price_uber =uber.groupby(by='time_segment')['price'].max()
print(Average_Price_uber)
print(Maximum_Price_uber)
print(Minimum_Price_uber)


# # Average, Minimum and Maximum Price during different time segments in a day for lyft

# In[85]:


Average_Price_uber =lyft.groupby(by='time_segment')['price'].mean()
Minimum_Price_uber =lyft.groupby(by='time_segment')['price'].min()
Maximum_Price_uber =lyft.groupby(by='time_segment')['price'].max()
print(Average_Price_uber)
print(Maximum_Price_uber)
print(Minimum_Price_uber)


# # What are the chances of having surge multiplier to the price

# In[86]:


lyft['surge_multiplier'].value_counts() 


# In[87]:


uber['surge_multiplier'].value_counts()


# In[88]:


#Surge multiplier distribution for lyft data 
sns.set(style="whitegrid")
plt.figure(figsize=(8,6))
total = float(len(lyft))
ax = sns.countplot(x="surge_multiplier", data=lyft)
plt.yscale('symlog')
plt.title('Surge Multiplier distribution for lyft ', fontsize=20)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='right')
plt.show()


# In[89]:


#Surge multiplier distribution for lyft data 
sns.set(style="whitegrid")
plt.figure(figsize=(8,6))
total = float(len(uber))
ax = sns.countplot(x="surge_multiplier", data=uber)
plt.yscale('symlog')
plt.title('Surge Multiplier distribution for uber', fontsize=20)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='right')
plt.show()


# Accordint to our dataset there is no surcharge added to the ubere cab fare. Where as for lyft data about 93.2% of the time, customers are not charged with surge. About 3.6% of time there is a 1.25x surge charge to the fare and so on . In this case I will multiply the surge multiplier to the initial estimated fare, to show the actual price for Lyft rides.

# In[90]:


#Effect of surcharge on weekdays , considering only dataset having surge multiplier greater than 1
weekday_surge = lyft[lyft["surge_multiplier"]> 1]
surge_charge = pd.DataFrame(weekday_surge.groupby(["Weekday", "surge_multiplier"]).size().reset_index())
surge_charge.columns = ["Weekday", "Surge Charge", "Count"]
plt.figure(figsize=(15, 5))
sns.barplot(x="Weekday", y="Count", hue="Surge Charge", data=surge_charge)
plt.title('Effect of surge charge on weekdays', fontsize=20)


# From the above bar graph we can see that the highest surge is happening on mondays and tuesdays.And wednesdays have low surge to the cab fare. We can also see that 1.25 surge is applied most of the times and 3 surcharge happens very rarely.

# In[91]:


x= pd.DataFrame(weekday_surge.groupby(["Weekday", "surge_multiplier","time_segment"]).size().reset_index())
x.columns = ["Weekday", "Surge Charge", "time_segment", "Count"]

plt.figure(figsize=(10, 8))
sns.barplot(x="time_segment", y="Count", hue="Surge Charge", data=x,ci=None)
plt.title('Effect of surge charge during the day', fontsize=20)


# According to the above bargraph we can see that surcharge is high during morning time and evenings , this could be because of lot of peple needing cab for work and school as morning and evening time is usually peak rush hour.Afternoon and night has the least Surge.

# # Do People Prefer Shared rides ? At what time segment during the day is it prefered?

# In[92]:


lyft_shared= lyft[lyft["name"] == "Shared"]
y = lyft_shared.groupby(["name", "time_segment"]).size().reset_index()
y.columns = ["Name", "time_segment", "Count"]
plt.figure(figsize=(15, 5))
sns.barplot(y="time_segment", x="Count",data=y, palette="pastel", orient ='h' )
plt.title('Distribution of shared cab through out the day', fontsize=20)


# # From the bar graph we can see that people prefer to take shared cabs during the morning and evening time . The reason for this could be the high rate of surge multiplied during this rush hour  

# # Top 3 Destinations by Uber and Lyft

# In[93]:


lyft_s_d_df= lyft.groupby(['source', "destination"]).size().reset_index()
lyft_s_d_df.columns = ["source", "destination", "count"]
lyft_s_d_df.sort_values("count", inplace=True, ascending = False)
lyft_five_most = lyft_s_d_df.iloc[0:5, ]
lyft_five_most["Source - Destination"] = lyft_s_d_df["source"] + " - " + lyft_s_d_df["destination"]

# So the top five most Source - Destination for lyft
lyft_five_most = lyft_five_most[["Source - Destination", "count"]]
lyft_five_most


# In[199]:


cab_types=uber['source'].value_counts()
cab_types


# In[95]:


c = ['pink', 'yellow', 'blue']
cab_types=uber['source'].value_counts()
cab_types.head(3).plot(kind='bar',color=c)
cab_types


# In[100]:


c = ['pink', 'yellow', 'blue']
cab_typeu=uber['destination'].value_counts()
cab_typeu.head(3).plot(kind='bar',color=c)
cab_typeu


# In[99]:



c = ['pink', 'yellow', 'blue']
cab_typel=lyft['destination'].value_counts()
cab_typel.head(3).plot(kind='bar',color=c)
cab_typel


# # Weekdays Where Most Rides were Taken 

# In[102]:


guber=sns.catplot(x='Weekday',kind='count',data=uber)
guber.set_xticklabels(rotation=90)
plt.show()


# In[105]:


guber=sns.catplot(x='Weekday',kind='count',data=lyft)
guber.set_xticklabels(rotation=90)
plt.show()


# # What is the effect of weather on the price and people prefference to take ride share?

# In[243]:


x=uber['icon'].value_counts()
x


# In[ ]:





# In[179]:


fig = px.bar(uber.head(5000),x='icon',y='price',color='icon', title = "Types of Lyft cabs and their price distribution",
               labels={
                     "price": "Price for each cab type",
                     "name": "Lyft cab type",
                 })
fig.show()


# In[ ]:


fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.show()


# In[184]:


dk.head()


# In[52]:


import plotly.express as px
fig = px.sunburst(df, path=['cab_type','name'], values='distance')

fig.show()


# In[80]:


from scipy.stats import chi2_contingency
def chi_square(c1,c2):
    chi_2, p_val, dof, exp_val = chi2_contingency(pd.crosstab(df[c1],df[c2],margins = False))# make sure margins = False
    print("Expected values: \n")
    print(exp_val)
    #print('\nChi-square is : %f'%chi_2, '\n\np_value is : %f'%p_val, '\n\ndegree of freedom is : %i'%dof)
    print(f'\nChi-square is : {chi_2}', f'\n\np_value is : {p_val}', f'\n\ndegree of freedom is :{dof}')

    if p_val < 0.05:# consider significan level is 5%
        print(F"\nThere is statistiacally significant correlation between {c1} and {c2} at 0.05 significant level")
    else:
        print(F"\nThere is no correlation between the two variables( we don't have enough evidence to conclude there is a a statistically significant relationship between {c1} and {c2}")  


# In[81]:


chi_square("price",'distance_segment') 


# In[ ]:




