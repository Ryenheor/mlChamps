import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingRegressor

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

def create_new_features(datas):
    features = ['maxPlayerLevel', 'numberOfAttemptedLevels',
       'attemptsOnTheHighestLevel', 'totalNumOfAttempts',
       'averageNumOfTurnsPerCompletedLevel', 'doReturnOnLowerLevels',
       'numberOfBoostersUsed', 'fractionOfUsefullBoosters', 'totalScore',
       'totalBonusScore', 'totalStarsCount', 'numberOfDaysActuallyPlayed']
    #в конце взять -
    #+0.0002
    #datas['custom_1'] = datas['maxPlayerLevel']/datas['numberOfAttemptedLevels']
    #-0.0001
    datas['custom_2'] = datas['attemptsOnTheHighestLevel']/datas['totalNumOfAttempts']
    #-0.0004
    datas['custom_3'] = datas['numberOfDaysActuallyPlayed']/datas['maxPlayerLevel']
    #+0.0001
    #datas['custom_4'] = datas['numberOfDaysActuallyPlayed']/datas['numberOfAttemptedLevels']
    #-0.0002
    datas['custom_6'] = datas['fractionOfUsefullBoosters']/datas['numberOfBoostersUsed']
    #+0.0012
    # datas['custom_7'] = datas['totalBonusScore']/datas['totalScore']
    #-0.0002
    datas['custom_8'] = datas['totalStarsCount']/datas['totalScore']
    #-0.0005
    datas['custom_81']= datas['totalStarsCount']/datas['totalBonusScore']
    #+0.0012
    temp_71 = datas['totalScore']+datas['totalBonusScore']
    #-0.0001
    datas['custom_81'] = datas['totalScore']/datas['totalStarsCount']
#   #
    #посчитать потом
    datas['allscoreofgame'] = datas['totalScore'] + datas['totalBonusScore']*temp_71 + datas['totalStarsCount']*datas['custom_81']
    #-0.0008
    datas['custom_9'] = datas['averageNumOfTurnsPerCompletedLevel']/datas['maxPlayerLevel']
    #-0.0001
    datas['custom_10'] = datas['attemptsOnTheHighestLevel']/datas['totalNumOfAttempts']
    #0.0000
    #datas["custom_12"] =datas['numberOfBoostersUsed']>datas['numberOfBoostersUsed'].mean()
#   #-0.0003
    #вывести в наружу до нормализации
    datas['custom_65'] = datas['numberOfBoostersUsed'] + datas['fractionOfUsefullBoosters']*datas['fractionOfUsefullBoosters']
    #-0.0004
    datas['custom_12']= datas['totalScore']/datas['maxPlayerLevel']
    #+0.0012
    #datas['bonus_in_overall'] = datas['totalBonusScore']/datas['totalScore']
    #-0.0001
    datas['score_per_day']= datas['totalScore']/datas['numberOfDaysActuallyPlayed']
#   #
    #посчитать потом
    datas['custom_19']= datas['allscoreofgame']/datas['numberOfDaysActuallyPlayed']
    #-0.0004
    datas['star_in_bonus']= datas['totalStarsCount']/(1 + datas['totalBonusScore'])

    #-0.0001
    datas['top_to_all'] = datas['attemptsOnTheHighestLevel']/datas['totalNumOfAttempts']
    #-0.0008
    datas['avg_attemps'] = datas['numberOfAttemptedLevels']/datas['totalNumOfAttempts']
    #-0.0002
    datas['custom_14']=datas['totalScore']/ datas['totalNumOfAttempts']
    #
    #посчитать потом
    #datas['custom_20']=datas['allscoreofgame']/ datas['totalNumOfAttempts']
    #+0.0007
    datas['custom_15']=datas['fractionOfUsefullBoosters']/ datas['totalNumOfAttempts']
    #-0.0011
    datas['custom_16']=datas['numberOfBoostersUsed']/ datas['totalNumOfAttempts']
    #-0.0008
    datas['custom_17']=datas['totalBonusScore']/ datas['totalNumOfAttempts']
    #-0.0006
    datas['custom_18']=datas['numberOfDaysActuallyPlayed']/ datas['totalNumOfAttempts']
    #-0.0003
    datas['custom_21'] = datas['totalNumOfAttempts']/datas['fractionOfUsefullBoosters']
    #+0.0002
    datas['custom_22'] = datas['attemptsOnTheHighestLevel']/datas['fractionOfUsefullBoosters']

def try_classivier(classifier, dataX,dataY):
    kf = KFold(dataX.shape[0],n_folds=2, shuffle=True)
    means=[]
    i=1
    for train, test in kf:
        classifier.fit(dataX[train], dataY[train])
        prediction = classifier.predict(dataX[test])
        curmean=log_loss(dataY[test],prediction )
        means.append(curmean)
        print(curmean)
    print("mean rezult: ".format(np.mean(means)))


data = pd.read_csv("x_train.csv", header=0, na_values='', delimiter=";", decimal='.')
test_data = pd.read_csv("x_test.csv", header=0, na_values='', delimiter=";", decimal='.')
y_train = pd.read_csv("y_train.csv", header=None, na_values='', delimiter=";")

print (data.shape)
print (y_train.shape)


data = (data - data.mean()) / data.std()

create_new_features(data)

X =pd.DataFrame(data, dtype=float)
Y = y_train #[0]

X=X.as_matrix()
Y=Y.as_matrix().ravel()

rf= GradientBoostingRegressor(n_estimators=69, max_features=24, min_samples_leaf=5)#()
#param_grid = [ {'n_estimators': list(range(5,500,1))}]
#rf = GridSearchCV(rf, param_grid=param_grid, cv=5)
rf.fit(X,Y)
prediction = rf.predict(X)

# print(rf.best_estimator_)
# print(rf.best_params_)
# print(rf.best_score_)
print (log_loss(Y,prediction))

try_classivier(rf,X,Y)

# test_data = (test_data - test_data.mean()) / test_data.std()
#
# create_new_features(test_data)
# X_test =pd.DataFrame(test_data, dtype=float)
# Y_test= rf.predict(X_test)
# df_to_save = pd.DataFrame(data={"res" : Y_test})
# df_to_save.to_csv("challenge2result_gbregr_1_56.csv", index=False, header=None, decimal='.')

