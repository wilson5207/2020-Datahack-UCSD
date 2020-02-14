#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Clean population csv

# In[3]:


population = pd.read_csv("SD1970_population.csv")

population = population.dropna()
population = population.drop(population[population["White persons"] == "..."].index)

TractName = population['Census Tract Name']
BlockGroup = population['Block Group']
Place_Name = population['Place Name']

population = population.drop('Census Tract Name',axis= 1)
population = population.drop('Block Group',axis= 1)
population = population.drop('Place Name',axis= 1)

for i in population.columns:
    #population[i] = population[i].str.replace(',',"").astype(int)
    population[i] = population[i].astype(str).str.replace(',',"")
    population[i] = population[i].replace(' ... ',"0")
    population[i] = population[i].replace('...','0')
    population[i] = population[i].astype(float)

        
population.insert(0, 'Place Name', Place_Name)
population.insert(0, 'Block Group', BlockGroup)
population.insert(0, 'Census Tract Name', TractName)

population.head()


# # Clean housing csv

# In[4]:


housing = pd.read_csv("SD1970_housing.csv")


housing = housing.dropna()
housing = housing.drop(housing[housing["Total owner occupied real $ aggregate (total) value of housing units"] == "..."].index)
tract_name = housing['Tract name']
housing = housing.drop("Tract name", axis= 1)
Block_group = housing['Block group']
housing = housing.drop("Block group", axis= 1)
place_name = housing['Place Name']
housing = housing.drop('Place Name',axis= 1)


for i in housing.columns:
    housing[i] = housing[i].astype(str).str.replace(',',"")
    housing[i] = housing[i].str.replace('$',"")
    housing[i] = housing[i].replace(' ... ',"0")
    housing[i] = housing[i].replace('...','0')
    housing[i] = housing[i].replace(' -   ',"0")
    housing[i] = housing[i].astype(float)
    
housing.insert(0, 'Place Name', place_name)
housing.insert(0, 'Block group', Block_group)
housing.insert(0, 'Tract name', tract_name)
housing.head()


# # What parameters do population.columns have?

# In[5]:


list(population.columns)


# # What parameters do housing.columns have?

# In[6]:


list(housing.columns)


# # What type of people does San Diego contain?

# In[7]:


San_diego = population.loc[population['Place Name'] == 'San Diego']
group_sd = San_diego.groupby(['Place Name']).sum()
list(group_sd.columns)
male_population_sd = group_sd[group_sd.columns[2:24]]
female_population_sd = group_sd[group_sd.columns[24:46]]

male_row = male_population_sd.iloc[0].values.tolist()
female_row = female_population_sd.iloc[0].values.tolist()
#Create a new table

indexs = ['<5', '5', '6', '7-9', '10-13', '14', '15', '16', '17', '18', '19', '20', '21', '22-24', '25-34', '35-44', '45-54', '55-59', '60-61', '62-64', '65-74', '75+']
male_population = pd.DataFrame(np.array([male_row]), columns = indexs)
female_population = pd.DataFrame(np.array([female_row]), columns = indexs)

print(male_population.plot.bar())
print(female_population.plot.bar())


# # Black people in San Diego?

# In[8]:



bmale_population_sd = group_sd[group_sd.columns[46:54]]
bfemale_population_sd = group_sd[group_sd.columns[54:62]]
bmale_row = bmale_population_sd.iloc[0].values.tolist()
bfemale_row = bfemale_population_sd.iloc[0].values.tolist()
combined_blacks_row = bmale_row + bfemale_row


demo_indexs = ['<5', '5-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65+']
bmale_population = pd.DataFrame(np.array([bmale_row]), columns = demo_indexs)
bfemale_population = pd.DataFrame(np.array([bfemale_row]), columns = demo_indexs)

print(bmale_population.plot.bar())
print(bfemale_population.plot.bar())


# # Nonwhites in San Diego?

# In[9]:


nwmale_population_sd = group_sd[group_sd.columns[62:70]]
nwfemale_population_sd = group_sd[group_sd.columns[70:78]]
nwmale_row = nwmale_population_sd.iloc[0].values.tolist()
nwfemale_row = nwfemale_population_sd.iloc[0].values.tolist()
combined_nw_row = nwmale_row + nwfemale_row

demo2_indexs = ['<5', '5-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65+']
nwmale_population = pd.DataFrame(np.array([nwmale_row]), columns = demo2_indexs)
nwfemale_population = pd.DataFrame(np.array([nwfemale_row]), columns = demo2_indexs)

print(nwmale_population.plot.bar())
print(nwfemale_population.plot.bar())


# In[10]:


list(group_sd.columns)[54:62]


# # Housing prices in San Diego

# In[11]:


list(housing.columns)


# In[12]:


housing_group = housing.copy()
tract_name = housing_group['Tract name']
housing_group = housing_group.drop("Tract name", axis= 1)
Block_group = housing_group['Block group']
housing_group = housing_group.drop("Block group", axis= 1)
housing_group = housing_group.groupby(['Place Name']).sum()

housing_group


# In[13]:


print(housing_group.plot.bar(y= 'Total owner occupied real $ aggregate (total) value of housing units'))
print(housing_group.plot.bar(y= 'Black owner occupied real $ aggregate (total) value of housing units'))


# In[14]:


condensed = housing[['Tract name', 'Block group', 'Place Name', 'Total housing units', 
        'Total owner occupied real $ aggregate (total) value of housing units', 
        'Black owner occupied real $ aggregate (total) value of housing units',]]
sd_condensed = condensed.loc[condensed['Place Name'] == 'San Diego']
sd_condensed = sd_condensed.groupby(['Block group']).sum()

print(sd_condensed.plot.bar(y = 'Total housing units'))
print(sd_condensed.plot.bar(y = 'Total owner occupied real $ aggregate (total) value of housing units'))
print(sd_condensed.plot.bar(y = 'Black owner occupied real $ aggregate (total) value of housing units'))


# In[15]:


sd_condensed['Black owner aggregate / Total owner'] = sd_condensed['Black owner occupied real $ aggregate (total) value of housing units'] / sd_condensed['Total owner occupied real $ aggregate (total) value of housing units']


# In[16]:


print(sd_condensed.plot.bar(y = 'Black owner aggregate / Total owner'))


# # A function to apply for other cities

# In[17]:


def ratio(n):
    housing_group = housing.copy()
    tract_name = housing_group['Tract name']
    housing_group = housing_group.drop("Tract name", axis= 1)
    Block_group = housing_group['Block group']
    housing_group = housing_group.drop("Block group", axis= 1)
    housing_group = housing_group.groupby(['Place Name']).sum()
    scondensed = housing[['Tract name', 'Block group', 'Place Name', 'Total housing units', 
        'Total owner occupied real $ aggregate (total) value of housing units', 
        'Black owner occupied real $ aggregate (total) value of housing units',]]
    city_condensed = scondensed.loc[condensed['Place Name'] == n]
    city_condensed = city_condensed.groupby(['Block group']).sum()
    city_condensed['Black owner aggregate / Total owner'] = city_condensed['Black owner occupied real $ aggregate (total) value of housing units'] / city_condensed['Total owner occupied real $ aggregate (total) value of housing units']
    print(city_condensed.plot.bar(y = 'Total housing units'))
    print(city_condensed.plot.bar(y = 'Total owner occupied real $ aggregate (total) value of housing units'))
    print(city_condensed.plot.bar(y = 'Black owner occupied real $ aggregate (total) value of housing units'))
    print(city_condensed.plot.bar(y = 'Black owner aggregate / Total owner'))

for i in housing_group.index:
    ratio(i)


# # Linear regression between Block group and Total aggregate?

# In[46]:


sd_housing_units_total_aggr = sd_condensed.drop('Black owner aggregate / Total owner', 
        axis=1).drop('Black owner occupied real $ aggregate (total) value of housing units', axis=1) # Clean the table
X = sd_housing_units_total_aggr.iloc[:, 0].values.reshape(-1, 1)  # Make each row into an array
Y = sd_housing_units_total_aggr.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()  # Create linear regression
linear_regressor.fit(X, Y)  # Perform linear regression
Y_pred = linear_regressor.predict(X)  # predictions


plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# # Correlation?

# In[48]:


sd_housing_units_total_aggr.corr(method ='pearson') 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




