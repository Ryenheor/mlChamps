import pandas as pd
import scipy as sp
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import  metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import matplotlib.pyplot as plt

plt.style.use('ggplot')


def create_new_features(datas):

    not_del = [96,165,71, 137, 200, 138, 11, 76, 208, 112, 182, 156,188,115, 131]#mlxtend
    for item in datas.columns:
        if item not in not_del:
            del datas[item]

    datas["x156"] = (datas[156] > 0.1)*1
    datas.loc[data[156] > 0.1 , 156] = 0.1

    datas["x11"] = sp.sqrt(datas[11]+0.2)#
    datas["x138"] = sp.around(datas[138],1)
    datas["x182"] = datas[182]*datas[182]
    datas["x96"] = (datas[96]>-0.03)*1
    print(len(datas.columns))


def try_xgb(dataX,dataY):
    kf = KFold(dataX.shape[0],n_folds=5, shuffle=True)
    means=[]
    for train, test in kf:
        rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=600, criterion="gini", max_features="sqrt", max_depth=20))
        rf.fit(dataX[train],dataY[train])
        predict_test = rf.predict(dataX[test])
        predict_test = sp.around(predict_test)
        actuals = dataY[test]
        means.append(metrics.accuracy_score(actuals, predict_test))
        print(metrics.accuracy_score(actuals, predict_test))
    print("mean rezult: {}".format(sp.mean(means)))
    print(str(means))

data = pd.read_csv("../x_train.csv", header=None, na_values='', delimiter=";", decimal='.')
test_data = pd.read_csv("../x_test.csv", header=None, na_values='', delimiter=";", decimal='.')
y_train = pd.read_csv("../y_train.csv", header=None, na_values='', delimiter=";")

# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train.astype("object"))
data_classes = label_encoder.transform(y_train.astype("object"))

create_new_features(data)
create_new_features(test_data)

data =pd.DataFrame(data, dtype=float)
test_data =pd.DataFrame(test_data, dtype=float)
datas_all = pd.concat((data, test_data))

data = data.as_matrix()
test_data = test_data.as_matrix()

# mlxtend
# rf = XGBRegressor()
# sfs1 = SequentialFeatureSelector(rf,
#        k_features=13,
#        forward=True,
#        floating=False,
#        verbose=2,
#        scoring='neg_mean_squared_error',
#        cv=5)
#
# sfs1 = sfs1.fit(data, data_classes)
#
# X_train_sfs = sfs1.transform(data)

try_xgb(data,data_classes)
rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=600, criterion="gini", max_features="sqrt", max_depth=20))#n_estimators=1000, criterion="gini", max_features="auto"

# parameters = {
#     "estimator__n_estimators": [500,1000,1500],
#     "estimator__max_features":["auto", "log2", "sqrt", None],
#     "estimator__criterion":["gini","entropy"],
#     "estimator__max_depth" : [None,5,6,8,10,12,13],
#     "estimator__oob_score" : [True, False],
#     "estimator__bootstrap" : [True, False]
# }
#rf = GridSearchCV(rf, param_grid=parameters, cv= 5, scoring='accuracy', verbose=11)
rf.fit(data,data_classes)
# print (rf.best_score_)
# print (rf.best_params_)
pred = rf.predict(test_data)
pred  = sp.around(pred)
#print(accuracy_score(y_train[0],pred))
df_to_save = pd.DataFrame(data={"res" : pred.astype(int)})
df_to_save.to_csv("mlch4_8_exp_rf14.csv", index=False, header=None, decimal='.')
