import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

class LassoRegressionModel:
    def __init__(self, model_name, a):
        self.__model = Pipeline([
            ("transformer", StandardScaler()),
            ("LinearModel", Lasso(alpha=a))])

    def __rmse_score(model, X, y):
        return np.sqrt(np.mean((y - self.model.predict(X)) ** 2))
    
    
    """
    function: train
    params: x: array like. ASSUMPTION: Assumes that X_training values are normalized
            Why Normalized? Lasso Regression is quite unkind of outliers, as a result,
            when dealing with different X parameters that could have different dimensions of
            variation.
            y: array like corresponding Y_training values, SHOULD NOT be normalized, but just
            original y array of values.
            
            Train determines which alpha value within a range set from .1 to 1.5 and test 60 steps
            minimizes alpha. You can minimize the range of alpha_arr values tested by changing
            the alpha_arr line
            For now is the rmse function added into the class handles scoring,however, this can be modified
            to a different scoring scheme, however, do be mindful of knowing if a different scheme 
            requires you to minimize or maximize the score and remember to alternate the min and max
            To be more certain of our training results, the function uses cross validation which outputs
            the mean of n different validation error runs which break up training set into a portion
            of training values and validation values to test training by seeing how predictions vary
            from validation values. does this cv = n times
          
    """
    def train(self, x, y):
        alpha_arr = np.linspace(0.1, 1.5, 60)
        cv_errors = []
        for alpha in alpha_arr:
            model.set_params(LinearModel__alpha=alpha)
            cv_error = np.mean(cross_val_score(model, x, u, scoring= self.__rmse_score, cv=5))
            cv_errors.append(cv_error)
         best_alpha_lasso = alpha_arr[cv_errors.index(min(cv_errors))]
         self.__model = Pipeline([
            ("transformer", StandardScaler()),
            ("LinearModel", Lasso(alpha=best_alpha_lasso))])
         self.__model.fit(x, y)

    def get_predictions(self, x):
        return np.round(self.__model.predict(x), 0).astype(np.int32)
