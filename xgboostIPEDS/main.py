from datetime import datetime
import numpy as np
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = np.loadtxt('./cleanDataNoHeading.csv', delimiter=',')

X = data[:, 1:13]  # columns 1 to 12 (skip 0 since it's just index)
y = data[:, 13]  # column 13

startTime = datetime.now()
sumTotal = 0
for i in range(1000):
    # split data into train = 75%, test = 25% of total data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # initialize and train model
    xgb_r = xg.XGBRegressor(n_estimators=10, max_depth=3, learning_rate=0.35)
    xgb_r.fit(X_train, y_train)

    # make predictions on test set
    y_pred = xgb_r.predict(X_test)

    # output Root Mean Squared Error for the model
    # print("RMSE:", metrics.mean_squared_error(y_test, y_pred, squared=False))

    sumTotal += metrics.mean_squared_error(y_test, y_pred, squared=False)

    if i % 100 == 0:
        print("iteration:", i)
print("Tweaked Parameters:")
print("average RMSE:", sumTotal/1000)
print("time used:", (datetime.now()-startTime).total_seconds(), "seconds")


startTime = datetime.now()
sumTotal = 0
for i in range(1000):
    # split data into train = 75%, test = 25% of total data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # initialize and train model
    xgb_r = xg.XGBRegressor()
    xgb_r.fit(X_train, y_train)

    # make predictions on test set
    y_pred = xgb_r.predict(X_test)

    # print("RMSE:", metrics.mean_squared_error(y_test, y_pred, squared=False))
    # print("R2 Score:", metrics.r2_score(y_test, y_pred))

    sumTotal += metrics.mean_squared_error(y_test, y_pred, squared=False)

    if i % 100 == 0:
        print("iteration:", i)
print("Default Parameters:")
print("average RMSE:", sumTotal/1000)
print("time used:", (datetime.now()-startTime).total_seconds(), "seconds")

# values are average RMSE over 1000 iterations of training and testing

# algorithm takes 10x longer with 10x estimators
# more than 10 estimators isn't worth the time
# RMSE with 10 estimators: 9.35%
# RMSE with 100 estimators: 9.32%

# RMSE with default max depth: 9.35%
# RMSE with 2 max depth: 9.11%
# RMSE with 3 max depth: 9.09%
# RMSE with 5 max depth: 9.23%
# RMSE with 10 max depth: 9.54%
# RMSE with 20 max depth: 9.58%

# RMSE with default learning rate: 9.09%
# RMSE with 0.2 learning rate: 11.27%
# RMSE with 0.3 learning rate: 9.10%
# RMSE with 0.35 learning rate: 9.01%
# RMSE with 0.4 learning rate: 9.09%
# RMSE with 0.5 learning rate: 9.31%
# RMSE with 0.7 learning rate: 9.86%
