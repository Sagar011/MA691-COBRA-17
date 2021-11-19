from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor

from sklearn.utils import shuffle
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import GridSearchCV

import math
import numpy as np
import random
import logging
import numbers


logger = logging.getLogger('pycobra.cobra')


class Cobra(BaseEstimator):


    def __init__(self, rndst=None, eps=None, machine_list='basic'):
        self.rndst = rndst
        self.eps = eps
        self.machine_list = machine_list
    
    def fit(self, X, y, default=True, X_k=None, X_l=None, y_k=None, y_l=None):

        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.X_k_ = X_k
        self.X_l_ = X_l
        self.y_k_ = y_k
        self.y_l_ = y_l
        self.estimators_ = {}
        if default:
            self.split_data()
            self.load_default(machine_list=self.machine_list)
            self.load_machine_predictions()

        return self


    def setEpsilon(self, X_eps=None, y_eps=None, grid_points=50):

        if self.eps is None and X_eps is not None:
            self.X_ = X_eps
            self.y_ = y_eps
            self.split_data()
            self.load_default()
            self.load_machine_predictions()
            a, size = sorted(self.all_predictions_), len(self.all_predictions_)
            res = [a[i + 1] - a[i] for i in range(size) if i+1 < size]
            emin = min(res)
            emax = max(a) - min(a)
            erange = np.linspace(emin, emax, grid_points)
            tuned_parameters = [{'eps': erange}]
            clf = GridSearchCV(self, tuned_parameters, scoring="neg_mean_squared_error")
            clf.fit(X_eps, y_eps)
            self.eps = clf.best_params_["eps"]
            self.estimators_, self.machine_predictions_ = {}, {}


    def pFun(self, X, alpha, info=False):

        select = {}
        for machine in self.estimators_:
            val = self.estimators_[machine].predict(X)
            select[machine] = set()
            for count in range(0, len(self.X_l_)):
                try:
                    if math.fabs(self.machine_predictions_[machine][count] - val) <= self.eps:
                        select[machine].add(count)
                except (ValueError, TypeError) as e:
                    logger.info("Error in indice selection")
                    continue

        points = []
        for count in range(0, len(self.X_l_)):

            row_check = 0
            for machine in select:
                if count in select[machine]:
                    row_check += 1
            if row_check == alpha:
                points.append(count)

        if len(points) == 0:
            if info:
                logger.info("No points were selected, prediction is 0")
                return (0, 0)
            return None

        avg = 0

        
        for point in points:
            avg += self.y_l_[point][1]
        avg = avg / len(points)

        if info:
            return avg, points
        return avg
    
    def pred_surv(self, X, alpha, info=False):

        select = {}
        for machine in self.estimators_:
            val = self.estimators_[machine].predict(X)
            select[machine] = set()
            for count in range(0, len(self.X_l_)):
                try:
                    if math.fabs(self.machine_predictions_[machine][count] - val) <= self.eps:
                        select[machine].add(count)
                except (ValueError, TypeError) as e:
                    logger.info("Error in indice selection")
                    continue

        points = []
        for count in range(0, len(self.X_l_)):
            row_check = 0
            for machine in select:
                if count in select[machine]:
                    row_check += 1
            if row_check == alpha:
                points.append(count)

        if len(points) == 0:
            if info:
                logger.info("No points were selected, prediction is 0")
                return (0, 0)
            return None

        avg_surv = None
        
        for point in points:
            for machine in select:
                val = self.estimators_[machine].predict_surv(self.X_l_[point].reshape(1,-1))[0]
                if avg_surv is None:
                    avg_surv = val
                else:
                    avg_surv += val
        avg_surv = avg_surv / avg_surv[0]

        if info:
            return avg_surv, points
        return avg_surv


    def predict(self, X, alpha=None, info=False):


        X = check_array(X)

        if alpha is None:
            alpha = len(self.estimators_)
        if X.ndim == 1:
            return self.pFun(X.reshape(1, -1), info=info, alpha=alpha)

        result = np.zeros(len(X))
        avg_points = 0
        index = 0
        for vector in X:
            if info:
                result[index], points = self.pFun(vector.reshape(1, -1), info=info, alpha=alpha)
                avg_points += len(points)
            else:
                result[index] = self.pFun(vector.reshape(1, -1), info=info, alpha=alpha)
            index += 1

        if info:
            avg_points = avg_points / len(X_array)
            return result, avg_points

        return result
    
    def predSurFun(self, X, alpha=None, info=False):

        X = check_array(X)

        if alpha is None:
            alpha = len(self.estimators_)
        if X.ndim == 1:
            return self.pred_surv(X.reshape(1, -1), info=info, alpha=alpha)

        result = [None]*len(X)
        avg_points = 0
        index = 0
        for vector in X:
            if info:
                result[index], points = self.pred_surv(vector.reshape(1, -1), info=info, alpha=alpha)
                avg_points += len(points)
            else:
                result[index] = self.pred_surv(vector.reshape(1, -1), info=info, alpha=alpha)
            index += 1

        result = np.array(result)
        
        if info:
            avg_points = avg_points / len(X_array)
            return result, avg_points

        return result

    def split_data(self, k=None, l=None, shuffle_data=False):


        if shuffle_data:
            self.X_, self.y_ = shuffle(self.X_, self.y_, rndst=self.rndst)

        if k is None and l is None:
            k = int(len(self.X_) / 2)
            l = int(len(self.X_))

        if k is not None and l is None:
            l = len(self.X_) - k

        if l is not None and k is None:
            k = len(self.X_) - l

        self.X_k_ = self.X_[:k]
        self.X_l_ = self.X_[k:l]
        self.y_k_ = self.y_[:k]
        self.y_l_ = self.y_[k:l]

        return self


    def load_default(self, machine_list='basic'):

        if machine_list == 'basic':
            machine_list = ['tree', 'ridge', 'random_forest', 'svm']
        if machine_list == 'advanced':
            machine_list=['lasso', 'tree', 'ridge', 'random_forest', 'svm', 'bayesian_ridge', 'sgd']

        self.estimators_ = {}
        for machine in machine_list:
            try:
                if machine == 'lasso':
                    self.estimators_['lasso'] = linear_model.LassoCV(rndst=self.rndst).fit(self.X_k_, self.y_k_)
                if machine == 'tree':
                    self.estimators_['tree'] = DecisionTreeRegressor(rndst=self.rndst).fit(self.X_k_, self.y_k_)
                if machine == 'ridge':
                    self.estimators_['ridge'] = linear_model.RidgeCV().fit(self.X_k_, self.y_k_)
                if machine == 'random_forest':
                    self.estimators_['random_forest'] = RandomForestRegressor(rndst=self.rndst).fit(self.X_k_, self.y_k_)
                if machine == 'svm':
                    self.estimators_['svm'] = LinearSVR(rndst=self.rndst).fit(self.X_k_, self.y_k_)
                if machine == 'sgd':
                    self.estimators_['sgd'] = linear_model.SGDRegressor(rndst=self.rndst).fit(self.X_k_, self.y_k_)
                if machine == 'bayesian_ridge':
                    self.estimators_['bayesian_ridge'] = linear_model.BayesianRidge().fit(self.X_k_, self.y_k_)
            except ValueError:
                continue
        return self


    def load_machine(self, machine_name, machine):

        self.estimators_[machine_name] = machine

        return self


    def load_machine_predictions(self, predictions=None):

        self.machine_predictions_ = {}
        self.all_predictions_ = np.array([])
        if predictions is None:
            for machine in self.estimators_:
                self.machine_predictions_[machine] = self.estimators_[machine].predict(self.X_l_)
                self.all_predictions_ = np.append(self.all_predictions_, self.machine_predictions_[machine])

        if predictions is not None:
            self.machine_predictions_ = predictions

        return self


