import numpy as np
import pandas as pd
import scipy as sp
from sklearn.cross_validation import KFold
from sklearn import metrics
from xgboost import XGBClassifier
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


def try_xgb(dataX,dataY):
    kf = KFold(dataX.shape[0],n_folds=5, shuffle=True)
    means=[]
    for train, test in kf:
        bst = XGBClassifier(max_depth=4,gamma=5, learning_rate=0.1, n_estimators=800)#
        bst.fit(dataX[train], dataY[train])
        predict_test = bst.predict_proba(dataX[test])
        actuals = dataY[test]
        means.append(metrics.log_loss(actuals, predict_test))
        print(metrics.log_loss(actuals, predict_test))
    print("mean rezult: {}".format(sp.mean(means)))
    print(str(means))

#mean rezult: 0.5394245574892765


def create_new_features(datas):

    datas.loc[datas["ap_hi"]  <0 , "ap_hi"] *= -1
    datas.loc[datas["ap_lo"]  <0 , "ap_lo"] *= -1

    datas.loc[datas["ap_hi"] >= 10000 , "ap_hi"] /=100

    datas.loc[datas["ap_hi"] >= 800 , "ap_hi"] /=10
    datas.loc[datas["ap_hi"] <= 30 , "ap_hi"] *= 10

    datas.loc[datas["ap_lo"] >= 4000 , "ap_lo"] /=100
    datas.loc[datas["ap_lo"] >= 300 , "ap_lo"] /= 10
    datas.loc[datas["ap_lo"] <= 10 , "ap_lo"] *= 10


    temp_ap_hi = datas["ap_hi"].copy(deep=True)
    temp_ap_lo = datas["ap_lo"].copy(deep=True)
    is_strange = datas["ap_hi"]<datas["ap_lo"]
    datas.loc[is_strange , "ap_hi"] =temp_ap_lo
    datas.loc[is_strange , "ap_lo"] =temp_ap_hi

    print(datas.head(10))

    radius = sp.sqrt(3*datas["weight"]/(1036*2*3.14*datas["height"]))*1000
    # # #
    datas["lenght_waist"] = 2*radius*3.14
    #  #
    datas["imt"] = datas["weight"]/ ((datas["height"]/100)*(datas["height"]/100))


    datas["norm_weight"] = pd.Series(np.nan)
    datas.loc[datas["gender"] == 2 , "norm_weight"] = (datas["height"]-110)*1.153
    datas.loc[datas["gender"] == 1 , "norm_weight"] = (datas["height"]-100)*1.153
    datas["x1"] = datas["weight"].divide(datas["norm_weight"])


    datas["ap_hi_norm"] = 102 + 0.6*(datas["gender"]/365)
    datas["ap_lo_norm"] = 63 + 0.5*(datas["gender"]/365)
    datas["x2"] = datas["ap_hi"].divide(datas["ap_hi_norm"])
    datas["x3"] = datas["ap_lo"].divide(datas["ap_lo_norm"])

    datas["x4"] = 109 + (0.5*(datas["age"]/365)) + (0.1 *datas["weight"])
    datas["x5"] = 63 + (0.1*(datas["age"]/365)) + (0.15 *datas["weight"])

    datas["x_1"] = datas["ap_hi"]-datas["ap_lo"]
    # imt & waist lenght

    del datas["ap_hi_norm"]
    del datas["ap_lo_norm"]

    #26 06 2017
    datas["MAD"] = datas["ap_lo"]+3.1*(datas["x_1"])
    del datas["x_1"]

    datas["x_2"] = datas["lenght_waist"]/(datas["age"]/365)

    datas["miller_norm_weight"] = pd.Series(np.nan)
    datas.loc[datas["gender"] == 2 , "miller_norm_weight"] = 56.12 + 1.41 * ((datas["height"]/2.54)-60)
    datas.loc[datas["gender"] == 1 , "miller_norm_weight"] = 53.1 + 1.36 * ((datas["height"]/2.54)-60)
    datas["x7"] = datas["weight"].divide(datas["miller_norm_weight"])


data = pd.read_csv("train.csv", header=0, na_values='None', delimiter=";", decimal='.')
test_data = pd.read_csv("test.csv", header=0, na_values='None', delimiter=";", decimal='.')

y_train = data["cardio"]

print(data.info())
print(test_data.info())

# ids =test_data["ID"]
# del data["ID"]
# del test_data["ID"]
# del data["id"]
# del test_data["id"]
del data["cardio"]

create_new_features(data)
create_new_features(test_data)

test_data = test_data.fillna(data.median(axis=0), axis=0)



# print(data.shape)
# data_alco = data.copy(deep=True)
# data_active = data.copy(deep=True)
# data_smoke = data.copy(deep=True)
#
# y_alco = data_alco['alco']
# y_active = data_active['active']
# y_smoke = data_smoke['smoke']
#
# del data_smoke['smoke']
# del data_alco['alco']
# del data_active['active']
#
# print(data.shape)
#
# print(data_active.shape)
# print(data_alco.shape)
# print(data_smoke.shape)
#
#
# #print(test_data.shape)
#
# class_alco = XGBClassifier(max_depth=4,gamma=5, learning_rate=0.1, n_estimators=800)
# class_alco.fit(data_alco.as_matrix(), y_alco.as_matrix())
#
# class_active = XGBClassifier(max_depth=4,gamma=5, learning_rate=0.1, n_estimators=800)
# class_active.fit(data_active.as_matrix(), y_active.as_matrix())
#
# class_smoke = XGBClassifier(max_depth=4,gamma=5, learning_rate=0.1, n_estimators=800)
# class_smoke.fit(data_smoke.as_matrix(), y_smoke.as_matrix())
#
#
# for index, row in test_data.iterrows():
#     row2 = row.copy(deep=True)
#     if math.isnan(row["alco"]):
#         row_alco = row2.copy(deep=True)
#         del row_alco["alco"]
#         test_data.set_value(index, 'alco', class_alco.predict_proba(row_alco)[:,1])
#     if math.isnan(row["active"]):
#         row_active = row2.copy(deep=True)
#         del row_active["active"]
#         test_data.set_value(index, 'active', class_active.predict_proba(row_active)[:,1])
#     if math.isnan(row["smoke"]):
#         row_smoke = row2.copy(deep=True)
#         del row_smoke["smoke"]
#         test_data.set_value(index, 'smoke', class_smoke.predict_proba(row_smoke)[:,1])
#
#

data =pd.DataFrame(data)
test_data =pd.DataFrame(test_data)
datas_all = pd.concat((data, test_data))

import xgboost as xgb
y_mean = np.mean(y_train)
# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 8000,
    'eta': 0.001,
    'max_depth': 5,
    'gamma':0.65,
    'subsample': 0.65,
    'min_child_weight': 2,
    'colsample_bytree': 0.6,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'nthread': 6,
    #'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(data, y_train)
dtest = xgb.DMatrix(test_data)
#
# xgboost, cross-validation
cv_result = xgb.cv(xgb_params,
                   dtrain,
                   num_boost_round=12000, # increase to have better results (~700)
                   early_stopping_rounds=100,
                   verbose_eval=50,
                   show_stdv=False
                  )
num_boost_rounds = len(cv_result)
print(num_boost_rounds)

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
prediction = model.predict(dtest)

# print(metrics.log_loss(y_train.as_matrix(),prediction))

df_to_save = pd.DataFrame(prediction)
df_to_save.to_csv("ch5_16_1.csv", index=False, decimal='.', sep=',', header=None )

