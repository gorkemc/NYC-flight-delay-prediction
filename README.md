# NYC-flight-delay-prediction

This project was prepared as part of the Machine Learning for Business Analytics course <a href="https://www.bu.edu/academics/questrom/courses/qst-ba-476/">MK 476</a> at Boston University in 2020. 
The aim of the project is to predict flight delay according to weather condition in New York City in 2013 by applying various machine learning techniques.

## Data
<img width="902" alt="NYCFlightsData" src="https://user-images.githubusercontent.com/45122094/140771252-bec7d4a3-700a-4d65-8d96-dfb51108fe8e.png">

The data is acquired from <a href="https://www.kaggle.com/aephidayatuloh/nyc-flights-2013">Kaggle</a>. It contains 19 columns and 336k rows.
Fro the weather a seperate dataset is used because it provided a weather classification for each hour, 
rather than raw numerical data such as temperature, humidity, barometric pressure which is hard to interpret.
This weather data set was aquired using Weather API on the <a href="https://openweathermap.org/apiwebsite">OpenWeatherMap </a> and is available under the ODbL License. 
Data is cleaned by omitting rows with null values and removing outliers. After na.omit function, the dataset is reduced to 127K rows of data, 
and after remowing the outliers by using quantile and iqr functions, the dataset reduced to 110K.

## Models
Forward stepwise selection is used for the subset selection. Final model included the folowing features:
weather, month, day, distance, sched_dep_time, sched_arr_time, air_time. For this model, 50000 rows are selected randomly for the test set. 
On this splitted data, several machine learning techniques are applied.  


### 1. Linear Regression
With the features decided, a linear regression model is applied. Mean quared error of this model corresponded to 87.6 which is approximately 9.4 minutes error margin.

```r
yhat.test.lm <- predict(fit.lm, dd.test)
mse.test.lm <- mean((y.test - yhat.test.lm)^2)
mse.test.lm
```

`## [1] 87.55104`

To gain better results regularization techniques are applied on this linear model.

### 2. Ridge Regression

As of L2 regularization, ridge regression performed similar output with slightly better mse score. Mean quared error of this model corresponded to 87.5 which is approximately 9.4 minutes error margin. 

```r
yhat.test.ridge <- predict(fit.ridge, x1.test, s = fit.ridge$lambda.min)
mse.test.ridge <- mean((y.test - yhat.test.ridge)^2)
mse.test.ridge
```

`## [1] 87.54174`

### 3. Lasso Regression

As of L1 regularization, lasso regression again performed similar output with slightly better mse score than linear regression but worse than ridge regression as expected. Mean quared error of this model corresponded to 87.5 which is approximately 9.4 minutes error margin. 

```r
yhat.test.lasso <- predict(fit.lasso, x1.test, s = fit.lasso$lambda.min)
mse.test.lasso <- mean((y.test - yhat.test.lasso)^2)
mse.test.lasso
```

`## [1] 87.54408`

So, all the regression models performed similarly with error margin of 9.4 minutes. Therefore, other machine learning techniques are applied to catch any improvement.

### 4. Decision Tree

The fitted decision tree performed slightly better than the regression models with mean squared error of 87.3 which is approximately 9.3 minutes error margin.

```r
yhat.tree.test <- predict(fit.tree, dd.test)
mse.tree.test <- mean((yhat.tree.test - y.test)^ 2)
mse.tree.test
```

`## [1] 87.33239`

### 5. Random Forest

For the random forest, the existing formula is used and the number of trees has changed several times. 
Even though the increase in the number of trees didn’t create a significant difference in the overall model, the test MSE score were better than regression models and decision tree.

<img width="754" alt="RandomForestTestError - NYCFlightData" src="https://user-images.githubusercontent.com/45122094/140769500-4f266de9-bd52-4485-9539-4a4f4af22ed6.png">

```r
yhat.rndfor.test <- predict(fit.rndfor, dd.test)
mse.rndfor.test <- mean((yhat.rndfor.test - y.test)^ 2)
mse.rndfor.test
```

`## [1] 86.96426`

### 6. Boosting
Again for the boosting, the existing formula is used. Different number of trees and shrinkage parameter is applied. The increase in the number of trees resulted in smaller test errors.

```r
# Plotting the test error vs number of trees
plot( n.trees , test.error,
pch=19, col="blue",
xlab="Number of Trees",
ylab="Test Error",
main = "Performance of Boosting on Test Set")
```
<img width="754" alt="BoostingTestErrors - NYCFlightData" src="https://user-images.githubusercontent.com/45122094/140769399-e1fd4dca-95cc-4a87-9866-a88c6863ad3d.png">

