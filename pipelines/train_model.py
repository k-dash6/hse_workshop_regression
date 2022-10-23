RS = 75

d = pd.DataFrame()
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
def metric(val_target, val_data, name, d = d): 

    a = r2_score(val_target, val_data)
    r = mean_squared_error(val_target, val_data)
    rmse = np.sqrt(r) 
    df = pd.DataFrame({"r2_score":([a]), "mean_squared_error":([r]), "rmse":([rmse])}, index=[name])
    return df

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
train_data, val_data, train_target, val_target = train_test_split(train, target, train_size=0.8, random_state= RS)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=RS)
for train_index, test_index in kf.split(train):
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    ridge = Ridge(random_state=RS).fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    a = metric(y_test, y_pred, "ridge")
    utils.save_as_pickle(metric(y_test, y_pred, 'ridge'), "../data/processed/metric_ridge.pkl")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor

LinearRegression = LinearRegression(n_jobs = -1)
GradientBoostingRegressor = GradientBoostingRegressor(random_state = RS)
CatBoost = CatBoostRegressor(iterations=100, loss_function='RMSE', eval_metric='RMSE', learning_rate=0.03, silent=True)

models = [LinearRegression, GradientBoostingRegressor, CatBoost]
names = ['LinearRegression', 'GradientBoostingRegressor', 'CatBoost']
for i in range(len(models)):
    a = models[i].fit(train_data, train_target).predict(val_data)
    utils.save_as_pickle(metric(val_target, a, names[i]), "../data/processed/metric_"+ names[i] +".pkl")
