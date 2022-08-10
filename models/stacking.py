from unicodedata import name
import numpy as np
import os
from pandas import test
from models.ensemble import Ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
from collections import defaultdict

class StackingClassifier(Ensemble):
    def __init__(self):
        super().__init__()
        self.weak_learners = {"dt": DecisionTreeClassifier(), 
                              "knn": KNeighborsClassifier(),
                              "rf": RandomForestClassifier(),
                              "gb": GradientBoostingClassifier()}
        self.final_learner = LogisticRegression()
    
        self.train_meta_model = None
        self.test_meta_model = None

        self.kf = KFold(n_splits=self.k, random_state=None)
        self.acc = defaultdict(list)
    
    def classifier(self):
        for name, clf in self.weak_learners.items():
            predictions_clf = self.k_fold_validation(name, clf)
            
            test_predict = self.train_model(clf)

            if isinstance(self.train_meta_model, np.ndarray):
                self.train_meta_model = np.vstack((self.train_meta_model, predictions_clf))
            
            else:
                self.train_meta_model = predictions_clf
            
            if isinstance(self.test_meta_model, np.ndarray):
                self.test_meta_model = np.vstack((self.test_meta_model, test_predict))
            else:
                self.test_meta_model = test_predict

        train_meta_model = self.train_meta_model.T

        test_meta_model = self.test_meta_model.T

        # Training stacking model
        self.train_stacking(self.final_learner, train_meta_model, test_meta_model)

    def k_fold_validation(self, name, clf):

        predictions_clf = None

        for train_index, test_index in self.kf.split(self.x_train):

            x_train, x_test = self.x_train[train_index], self.x_train[test_index]
            y_train, y_test = self.y_train[train_index], self.y_train[test_index]
            
            clf.fit(x_train, y_train)
            prediction = clf.predict(x_test)

            acc = accuracy_score(prediction , y_test)
            
            print(f"the accracy of {name} is {acc}")
            self.acc[name] = acc

            if isinstance(predictions_clf, np.ndarray):
                predictions_clf = np.concatenate((predictions_clf, prediction))
            else:
                predictions_clf = prediction
        return predictions_clf

    def train_model(self, clf):

        clf.fit(self.x_train, self.y_train)

        y_pred = clf.predict(self.x_test)

        return y_pred

    def train_stacking(self, final_learner, train_meta_model, test_meta_model):

        final_learner.fit(train_meta_model, self.y_train)

        print(f"Train accuracy: {final_learner.score(train_meta_model, self.y_train)}")
        print(f"Test accuracy: {final_learner.score(test_meta_model, self.y_test)}")

        self.acc["final"] = final_learner.score(train_meta_model, self.y_train)