# TMDB BOX OFFICE PREDICTION

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

## Table of Contents
  - Introduction
  - Data Processing
  - Exploratory Data Analysis
  - Model Building and selection
  - Hyperparameters Optimization

### Introduction
In this project, we had metadata on over 7,000 past films from [The Movie Database] to try and predict the overall worldwide box office revenue. The objective is to build regression models and to optimize the model for a better prediction accuracy with ML/DL methods. This project will help to discover customer preferences and give guidance to entertainment companies.

### Data Processing 
Data points provided include cast, crew, plot keywords, budget, release dates, languages, production companies, and countries.
In this part:
  - Cut outliers, fill missing values, use log to process smooth skewed features
  - Use One Hot Encoder, Label Encoder and Vectorization in text processing
  - Join text variables and create vector matrix with TF-IDF and find relations between text words with Global Vectors for Word Representation
 
[Global Vectors for Word Representation] is unsupervised learning algorithm for obtaining vector representations for words. 

### Exploratory Data Analysis
Provided in-depth Exploratory Data Analysis (EDA) with variables such as budget, runtime, release dates, number of languages, production companies, and countries and explore their relations and impacts on revenue. 

### Modeling
- Build prediction models including XGBoost, LightGBM and check weights of feature importance with LGBMRegressor
- Evaluate the performance of models with Root Mean Squared Error (RMSE)

### Hyperparameters Optimization
- Used Random Grid Search and Bayesian Optimization to find the best parameters for the better accuracy

For example, try random and grid search:

```sh
from sklearn.model_selection import RandomizedSearchCV
param_dist = {"learning_rate": uniform(0, 1),
              "gamma": uniform(0, 5),
              "max_depth": range(1,50),
              "n_estimators": range(1,300),
              "min_child_weight": range(1,10)}
rs = RandomizedSearchCV(xgb, param_distributions=param_dist, 
                        scoring='neg_mean_squared_error', n_iter=25)
```

### Some of findings

- Since we have 252 features, to check each weight of feature, we choose LGBMRegressor which is a gradient boosting algorithm based on decision tree.

Most important features:
| Weight | Feature |
| ------ | ------ |
| 0.4473 | budget |
| 0.1399 | popularity|
| 0.0625 | release_date_year |
| 0.0265 | runtime |
| 0.0235 | genres|
| 0.0152 | collection_name |

- In parameter tunning, the better solution is Bayesian optimization. With this method in this project, the number of iterations is small and the speed is fast. 
> Bayesian optimisation chooses the next hyperparameters 
> in an informed way and as such spends more time evaluating 
> areas of the parameter distribution it believes have the highest 
chance of bringing a cross-validation score improvement 
versus previous iterations.
> This can result in fewer evaluations of the objective function
and better generalisation performance on the test set 
compared to random or grid search.

**Happy coding!**



   [The Movie Database]: <https://www.themoviedb.org/?language=en-US>
   [Global Vectors for Word Representation]: <https://nlp.stanford.edu/pubs/glove.pdf>
   
