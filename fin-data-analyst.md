# **Self training work on data consepts using GDP and life satisfaction**
## The aim of this project is to demonstrate data analysis concepts and to test some hypothesis in this data. 
### The Data was imported from github.com  . Below are the various pakages use in this experiment. The csv file was save as lifesat.


```python
### loading the necessary packages for the entire project. 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.formula.api import ols 
from scipy.stats import norm

import sklearn 

from sklearn.linear_model import LinearRegression
```


```python
### importing the data set from an online data base and saving the data set as lifesat 

data_root=   "https://github.com/ageron/data/raw/main/"
lifesat= pd.read_csv(data_root + "lifesat/lifesat.csv")
lifesat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>GDP per capita (USD)</th>
      <th>Life satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Russia</td>
      <td>26456.387938</td>
      <td>5.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Greece</td>
      <td>27287.083401</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Turkey</td>
      <td>28384.987785</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Latvia</td>
      <td>29932.493910</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hungary</td>
      <td>31007.768407</td>
      <td>5.6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Portugal</td>
      <td>32181.154537</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Poland</td>
      <td>32238.157259</td>
      <td>6.1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Estonia</td>
      <td>35638.421351</td>
      <td>5.7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Spain</td>
      <td>36215.447591</td>
      <td>6.3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Slovenia</td>
      <td>36547.738956</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Lithuania</td>
      <td>36732.034744</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Israel</td>
      <td>38341.307570</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Italy</td>
      <td>38992.148381</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>United Kingdom</td>
      <td>41627.129269</td>
      <td>6.8</td>
    </tr>
    <tr>
      <th>14</th>
      <td>France</td>
      <td>42025.617373</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>New Zealand</td>
      <td>42404.393738</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Canada</td>
      <td>45856.625626</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Finland</td>
      <td>47260.800458</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Belgium</td>
      <td>48210.033111</td>
      <td>6.9</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Australia</td>
      <td>48697.837028</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Sweden</td>
      <td>50683.323510</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Germany</td>
      <td>50922.358023</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Austria</td>
      <td>51935.603862</td>
      <td>7.1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Iceland</td>
      <td>52279.728851</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Netherlands</td>
      <td>54209.563836</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Denmark</td>
      <td>55938.212809</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>26</th>
      <td>United States</td>
      <td>60235.728492</td>
      <td>6.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Data Cleaning. 
## the data is good for analysis but to ease our analysis the headings were simplified. 
```


```python
## Data cleaning
lifesat
life_sat=lifesat.rename(columns={'GDP per capita (USD)':'GDP', 'Life satisfaction':'life'})
```


```python
x= lifesat[['GDP per capita (USD)']].values
x
y= lifesat[['Life satisfaction']]
```

# Understanding the relationship between life satisfaction and gdp per capital
## We noticed from the graph below that as GDp per capital increase so does life satisfation too.
## Implying countries should keep improving  gdp per capital 


```python
lifesat.plot(kind='scatter',grid=True, x= 'GDP per capita (USD)',
             y =  'Life satisfaction')
plt.axis([23_500, 62_500, 4,9])
plt.show()


```


    
![png](output_7_0.png)
    


# In the following stages , varoiuse statistics test were performed to understand 
## Sampling
### Random sampling.
### Finding the population mean and sample mean of the data set. There was no significants difference between the sample means and popultaion means for life expectancy.  mean was given as 5.6


```python
# Statistic testin using various statistic techniques
## perform some sampling technics on the life state data.
lifsat_pop= life_sat[['GDP', 'life']]
```


```python
### understanding the data before performing statistical test
lifsat_pop.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GDP</th>
      <th>life</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26456.387938</td>
      <td>5.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27287.083401</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28384.987785</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29932.493910</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31007.768407</td>
      <td>5.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
##.1  samplelifsat using random statistic(understanding randomstatistics)
lifesat_sample= lifsat_pop.sample(n=10)
lifesat_sample
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GDP</th>
      <th>life</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>55938.212809</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>18</th>
      <td>48210.033111</td>
      <td>6.9</td>
    </tr>
    <tr>
      <th>19</th>
      <td>48697.837028</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>42025.617373</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>45856.625626</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>35638.421351</td>
      <td>5.7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>32181.154537</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>38992.148381</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>32238.157259</td>
      <td>6.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31007.768407</td>
      <td>5.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
## check population paramater
### mean using numpy
import numpy as np
np.mean(lifsat_pop['life'])
```




    6.566666666666666




```python
lifsat_pop['life'].mean()

```




    6.566666666666666



## Does life expectancy follows a normal distribution. 
### plotting both the population and the sample data set shows that the distribution is not normally distributed. 


```python
###  visualising  GDP
### Both pop data and sample data
lifsat_pop['life'].hist(bins=np.arange(2,9,0.5))
lifesat_sample['life'].hist(bins=np.arange(2,9,0.5))
```




    <AxesSubplot:>




    
![png](output_15_1.png)
    


# This random number sampling is used to demonstrate a normal curve distribution. 


```python
# visualised a random number sample to understand if the plot above follows a simple random distribution grap
random = np.random.beta(a=10, b=10, size=5000)
random
plt.hist(random, bins=np.arange(0,1,.05))
```




    (array([  0.,   0.,   2.,   8.,  27., 123., 281., 467., 730., 840., 949.,
            701., 458., 262., 108.,  36.,   7.,   1.,   0.]),
     array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
            0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95]),
     <BarContainer object of 19 artists>)




    
![png](output_17_1.png)
    



```python
### Performing random sampling on the data set

life_sat.sample(n=5, random_state=19000113)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>GDP</th>
      <th>life</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Latvia</td>
      <td>29932.493910</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Canada</td>
      <td>45856.625626</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Australia</td>
      <td>48697.837028</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Spain</td>
      <td>36215.447591</td>
      <td>6.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Turkey</td>
      <td>28384.987785</td>
      <td>5.5</td>
    </tr>
  </tbody>
</table>
</div>



## Understanding systematic sampling. 



```python
### understanding systematic sample
Sample_size= 5
pop_size= len(life_sat)
pop_size
```




    27




```python
interval = pop_size//Sample_size
interval
```




    5




```python
## systematic sampling
```


```python
### systematic samplin  and rol selection
life_system= life_sat.iloc[::interval]
life_system
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>GDP</th>
      <th>life</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Russia</td>
      <td>26456.387938</td>
      <td>5.8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Portugal</td>
      <td>32181.154537</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Lithuania</td>
      <td>36732.034744</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>15</th>
      <td>New Zealand</td>
      <td>42404.393738</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Sweden</td>
      <td>50683.323510</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Denmark</td>
      <td>55938.212809</td>
      <td>7.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
### ploting systematic sampling using reset data set call life_system_Id.
life_system_id= life_system.reset_index()
life_system_id
life_system_id.plot(x="index", y='GDP', kind="scatter")
plt.show()
```


    
![png](output_24_0.png)
    



```python
life_system_id
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Country</th>
      <th>GDP</th>
      <th>life</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Russia</td>
      <td>26456.387938</td>
      <td>5.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Portugal</td>
      <td>32181.154537</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>Lithuania</td>
      <td>36732.034744</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>New Zealand</td>
      <td>42404.393738</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>Sweden</td>
      <td>50683.323510</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25</td>
      <td>Denmark</td>
      <td>55938.212809</td>
      <td>7.6</td>
    </tr>
  </tbody>
</table>
</div>



# **Hypothesis testing.**
### What is the implication of mean GDP greater than 5000
### We start by  samplig the Data set. 


```python
shuffled= life_sat.sample(frac=1)
shuffled= shuffled.reset_index(drop= True).reset_index()
shuffled
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Country</th>
      <th>GDP</th>
      <th>life</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Turkey</td>
      <td>28384.987785</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Denmark</td>
      <td>55938.212809</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Spain</td>
      <td>36215.447591</td>
      <td>6.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Latvia</td>
      <td>29932.493910</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Lithuania</td>
      <td>36732.034744</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Austria</td>
      <td>51935.603862</td>
      <td>7.1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>New Zealand</td>
      <td>42404.393738</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>Italy</td>
      <td>38992.148381</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>Poland</td>
      <td>32238.157259</td>
      <td>6.1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>Estonia</td>
      <td>35638.421351</td>
      <td>5.7</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>Netherlands</td>
      <td>54209.563836</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>Sweden</td>
      <td>50683.323510</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>Belgium</td>
      <td>48210.033111</td>
      <td>6.9</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>Greece</td>
      <td>27287.083401</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>Australia</td>
      <td>48697.837028</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>France</td>
      <td>42025.617373</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>Russia</td>
      <td>26456.387938</td>
      <td>5.8</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>Hungary</td>
      <td>31007.768407</td>
      <td>5.6</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>Iceland</td>
      <td>52279.728851</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>Canada</td>
      <td>45856.625626</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>United States</td>
      <td>60235.728492</td>
      <td>6.9</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>Israel</td>
      <td>38341.307570</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>Slovenia</td>
      <td>36547.738956</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>Portugal</td>
      <td>32181.154537</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>Germany</td>
      <td>50922.358023</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>Finland</td>
      <td>47260.800458</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>United Kingdom</td>
      <td>41627.129269</td>
      <td>6.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
shuffled.plot(x="index", y= 'GDP', kind= "scatter")
```




    <AxesSubplot:xlabel='index', ylabel='GDP'>




    
![png](output_28_1.png)
    


## next we calculate the relative errors in the sample distribution
the relative errors was given as 13.922


```python
# calculating relative error
# 100*(pop_mean-sample_mean)/pop_mean

life_sat_mean= life_sat['GDP'].mean()
life_sat_mean
life_sat_sample_mean= life_sat.sample(n=10, )['GDP'].mean()
life_sat_sample_mean
reletive_error= 100*abs(life_sat_mean-life_sat_sample_mean)/life_sat_mean
reletive_error
```




    13.922817390655053




```python
# sample sizes
sample_sizes= len( life_sat.sample(n=10, ))
sample_sizes
```




    10



 ## A boodstrap distribution of the error term.

 Finding the standard error term use in calculating Z-score


```python
# get the error of the standard distribution and means

mean_cup_points_1000=[]
for i in range (1000):
    mean_cup_points_1000.append(
       np.mean(life_sat.sample(frac=1,replace=True)['GDP'])
    )
boodstrap_distn=mean_cup_points_1000
```


```python
plt.hist(boodstrap_distn, bins=15)
plt.show()

```


    
![png](output_34_0.png)
    



```python
# the standard error of 
std_erre= np.std(boodstrap_distn, ddof=1)
```

# calculating the population mean .
The population mean was given as 41564. From this means it can clarly be stated that a GDP of 5000 and above signifies an average above the countries average GDP.
This is an indicator that an economy with a gdp above 5000 economy performs better than it peers. However we most test the significanse of our result using Z-score. 


```python
# mean of life_sat
mean = life_sat['GDP'].mean()
mean
```




    41564.521771015454




```python

```


```python
# gdp greater than 5000 shows a better performing economics
GDP_5000= 5000
Z= (5000-mean)/std_erre
Z
```




    -19.461721035916252


