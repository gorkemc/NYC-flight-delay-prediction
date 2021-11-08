# NYC-flight-delay-prediction

This project was prepared as part of the Machine Learning for Business Analytics course MK 476 at Boston University. 
The aim of the project is to predict flight delay according to weather condition in New York City in 2013 by applying various machine learning techniques.

## Data

The data is acquired from <a href="https://www.kaggle.com/aephidayatuloh/nyc-flights-2013">Kaggle</a>. It contains 19 columns and 336k rows.
Fro the weather a seperate dataset is used because it provided a weather classification for each hour, 
rather than raw numerical data such as temperature, humidity, barometric pressure which is hard to interpret.
This weather data set was aquired using Weather API on the <a href="https://openweathermap.org/apiwebsite">OpenWeatherMap </a> and is available under the ODbL License. 
Data is cleaned by omitting rows with null values and removing outliers. After na.omit function, the dataset is reduced to 127K rows of data, 
and after remowing the outliers by using quantile and iqr functions, the dataset reduced to 110K.

## Models
Forward stepwise selection is used for the subset selection and on this subset 50000 rows are selected randomly for the test set. 
On this splitted data, several machine learning techniques are applied.  

###Linear Regression
