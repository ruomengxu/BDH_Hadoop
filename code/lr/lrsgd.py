#!/usr/bin/env python
"""
Implement your own version of logistic regression with stochastic
gradient descent.

Author: Meera Kamath
Email : mkamath6@gatech.edu
"""

import collections
import math


class LogisticRegressionSGD:

    def __init__(self, eta, mu, n_feature):
        """
        Initialization of model parameters
        """
        self.eta = eta
        self.weight = [0.0] * n_feature
        self.mu = mu

    def fit(self, X, y):
        """
        Update model using a pair of training sample
        """
        n_feature = len(self.weight)
        p = self.predict_prob(X)
        for i in range(n_feature):
            self.weight[i] = (1-2*self.mu*self.eta)*self.weight[i]
        for i in range(len(X)):
            self.weight[X[i][0]] = self.weight[X[i][0]]+self.eta*(y - p)*X[i][1]

    def predict(self, X):
        return 1 if self.predict_prob(X) > 0.5 else 0

    def predict_prob(self, X):
        return 1.0 / (1.0 + math.exp(-math.fsum((self.weight[f]*v for f, v in X))))
