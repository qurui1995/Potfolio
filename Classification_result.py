import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import  OneHotEncoder
import scipy
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import datetime as dt
import math
import lightgbm as lgb
from  sklearn.utils import compute_class_weight
from lightgbm import LGBMClassifier
import xgboost as xgb
import itertools
# from catboost import CatBoostClassifier

# import catboost

import plotly 
import plotly.express as px
# import plotly.graph_objs as go
# import plotly.offline as py
# from plotly.offline import iplot
# from plotly.subplots import make_subplots
# import plotly.figure_factory as ff

# import shap 
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import roc_auc_score, classification_report, auc, roc_curve,f1_score, confusion_matrix,precision_recall_curve,log_loss,precision_score, accuracy_score
import pandas_profiling as pp

from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from optuna.integration import XGBoostPruningCallback


def check_result(estimator, X_train, y_train, X_test, y_test, metric = 'gmeans'):
    clf = estimator.fit(X_train, y_train)

    print(confusion_matrix(y_test, clf.predict(X_test)))
    print('---------------------Train------------------------------\n')
    print(classification_report(y_train, clf.predict(X_train)))
    print('---------------------Test------------------------------\n')
    print(classification_report(y_test, clf.predict(X_test)))
    print('---------------------Train------------------------------\n')
    fpr, tpr, thresholds = roc_curve(y_train, clf.predict_proba(X_train)[:, 1])
    plt.figure(1)
    plt.plot([0,1], [0,1], 'k--', label='No skill (AUC = 0.50)')
    plt.plot(fpr, tpr, label='Elastic Net Classifier (AUC = %0.3f)' % (auc(fpr, tpr)))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()
    print('---------------------Test------------------------------\n')
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # plot the roc curve for the model

    plt.figure(1)
    plt.plot([0,1], [0,1], 'k--', label='No skill (AUC = 0.50)')
    plt.plot(fpr, tpr, label='Elastic Net Classifier (AUC = %0.3f)' % (auc(fpr, tpr)))
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')


    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()
    
    print('---------------------Train------------------------------\n')
    print(classification_report(y_train, (clf.predict_proba(X_train)[:,1]>=thresholds[ix]).astype(int)))  
    print(confusion_matrix(y_train, (clf.predict_proba(X_train)[:,1]>=thresholds[ix]).astype(int)))
    print('---------------------Test------------------------------\n')
    print(classification_report(y_test, (clf.predict_proba(X_test)[:,1]>=thresholds[ix]).astype(int)))
    print(confusion_matrix(y_test, (clf.predict_proba(X_test)[:,1]>=thresholds[ix]).astype(int)))
