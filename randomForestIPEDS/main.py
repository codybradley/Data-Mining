from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

    regr = RandomForestRegressor(n_estimators=100, max_features="sqrt", max_depth=20)
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    sumTotal += metrics.mean_squared_error(y_test, y_pred, squared=False)
    if i % 100 == 0:
        print("Iteration:", i)
print("Tweaked Parameters")
print("average RMSE:", sumTotal/1000)
print("time used:", (datetime.now()-startTime).total_seconds())

startTime = datetime.now()
sumTotal = 0
for i in range(1000):
    # split data into train = 75%, test = 25% of total data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # initialize and train model
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)

    # make a prediction for each test sample
    y_pred = regr.predict(X_test)

    sumTotal += metrics.mean_squared_error(y_test, y_pred, squared=False)
    if i % 100 == 0:
        print("Iteration:", i)
print("Default Parameters")
print("average RMSE:", sumTotal/1000)
print("time used:", (datetime.now()-startTime).total_seconds())

# average RMSE with no parameters tweaked: 8.52%

# negligible difference in RMSE for n_estimators,
# but 10x estimators takes 10x time (use less)
# average RMSE with 100 estimators = 8.52%
# average RMSE with 1000 estimators = 8.48%

# sqrt max features better than auto max features
# average RMSE with auto max features = 8.52%
# average RMSE with sqrt max features = 8.40%

# improvement in RMSE falls off after about 20 max depth
# average RMSE with 3 max depth = 9.16%
# average RMSE with 5 max depth = 8.59%
# average RMSE with 10 max depth = 8.40%
# average RMSE with 20 max depth = 8.36%

# average RMSE after tweaking parameters: 8.36%
