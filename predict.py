"""
Ref: https://www.kaggle.com/the1owl/surprise-me
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from time import time
from datetime import datetime
import matplotlib.pyplot as plt

def visualize(train_df, test_df):    
    ts_train = train_df.groupby(['visit_date'])[['visitors']].sum()
    ts_pred = test_df.groupby(['visit_date'])[['visitors']].sum()
    plt.rcParams["figure.figsize"] = (16,4)
    plt.plot(ts_train)
    plt.plot(ts_pred)
    plt.savefig('xgb_result1.jpg')
    
    ts_train1 = train_df[:391][['visit_date', 'visitors']].set_index('visit_date')
    ts_pred1 = test_df[23361:23399][['visit_date', 'visitors']].set_index('visit_date')
    plt.plot(ts_train1)
    plt.plot(ts_pred1)
    plt.savefig('xgb_result2.jpg')

train_df = pd.read_csv('data/train_df.csv', parse_dates=['visit_date'])
test_df = pd.read_csv('data/test_df.csv', parse_dates=['visit_date'])

train_x = train_df.drop(['air_store_id', 'visit_date', 'visitors'], axis=1)
train_y = np.log1p(train_df['visitors'].values)
print(train_x.shape, train_y.shape)
test_x = test_df.drop(['id', 'air_store_id', 'visit_date', 'visitors'], axis=1)

#boost_params = {'eval_metric': 'rmse'} #, 'gpu_id':0, 'tree_method':'gpu_hist'}
xgb0 = xgb.XGBRegressor(
    max_depth=8,
    learning_rate=0.01,
    n_estimators=10000,
    objective='reg:linear',
    gamma=0,
    min_child_weight=1,
    subsample=1,
    colsample_bytree=1,
    scale_pos_weight=1,
    seed=27,
    eval_metric='rmse',
    tree_method='gpu_hist',
    predictor ='gpu_predictor',    
    gpu_id=0
)
    #**boost_params)

t0 = time()
xgb0.fit(train_x, train_y)
print("done in %0.3fs" % (time() - t0))
predict_y = xgb0.predict(test_x)
test_df['visitors'] = np.expm1(predict_y)
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filepath_prediction = 'prediction/prediction-%s.csv' % timestamp
test_df[['id', 'visitors']].to_csv(filepath_prediction, index=False, float_format='%.3f')  # LB0.495
print('Prediction saved to %s' % filepath_prediction)
visualize(train_df, test_df)