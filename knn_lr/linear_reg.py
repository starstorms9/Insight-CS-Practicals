'''
Save file as: FIRSTNAME_LASTINITIAL_LR.py
Timestamp: 02042020
Objective: Coding Linear regression from scratch
Dependcies: Python 3.6, numpy

Last 45 MINUTES
@author: Amber

Linear Regression
Resources
https://towardsdatascience.com/linear-regression-using-python-ce21aa90ade6
https://www.kaggle.com/aariyan101/usa-housingcsv
** For the practical today **:
download this : https://www.kaggle.com/aariyan101/usa-housingcsv

**Question**:
- Train linear regression model on above data

** Background **:
- Linear Regression is a way of predicting a response Y on the basis of a single predictor variable X.
 It is assumed that there is approximately a linear relationship between X and Y.
 Mathematically, we can represent this relationship as:
- Y ≈ ɒ + ß X + ℇ
- where ɒ and ß are two unknown constants that represent intercept and slope terms in the linear model
and ℇ is the error in the estimation.

** Steps **:
- start by taking the simplest possible example. Calculate the regression with only two data points
    (price for dependent and number of rooms for independent)
- then use price as the dependent variable and all others as independent variables

** Outputs **
  - Function for train/test splitting
  - Function for fitting the model
  - Function for running prediction on the holdout (test) set
  - Function for outputting some analysis (organized text here is fine)
  
**BONUS STEPS**:
- Explore the correlation using Pearson Correlation Coefficient. If you want resources for this check out the [pandas implementation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html) or the [Wiki article](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
- Want to test your visualization mettle? Re-implement Seaborn's  [library for plotting pairwise relationships in a dataset](https://seaborn.pydata.org/generated/seaborn.pairplot.html)
'''
#%% Imports
import numpy as np
import pandas as pd

#%% Solution
class linReg() :
    def __init__() :
        pass
    
    def fit(self) :
        pass
    
    
def loadData() :
    data = pd.read_csv('USA_Housing.csv')
    return data


#%% Testing
data = loadData() 