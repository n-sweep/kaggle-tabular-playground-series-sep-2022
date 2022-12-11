# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

# %%
sub = pd.read_csv('data/sample_submission.csv')
test = pd.read_csv('data/test.csv')
train = pd.read_csv('data/train.csv')

# %%
sub.head()

# %%
test.head()

# %%
train.head()

# %%
train['country'].unique()

# %%
train['store'].unique()

# %%
train['product'].unique()

# %%
train['date'].unique()

# %%
train.head()

# %%
train.drop('row_id', inplace=True, axis=1)
test.drop('row_id', inplace=True, axis=1)
train["country"] = train["country"].astype("category")
train["store"] = train["store"].astype("category")
train["product"] = train["product"].astype("category")

# %%
tscv = TimeSeriesSplit()
tscv

# %%
y, X = train['num_sold'], train.drop('num_sold', axis=1)

# %%
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    break

# %%
X_train

# %%
X_train = pd.get_dummies(X_train, columns=['country', 'store', 'product'])
X_test = pd.get_dummies(X_test, columns=['country', 'store', 'product'])

# %%
X_train.info()

# %%
X_train['date'] = pd.to_datetime(X_train['date']).astype('int64') // 10 ** 9
X_test['date'] = pd.to_datetime(X_test['date']).astype('int64') // 10 ** 9

# %%
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
param = {
    'max_depth': 2,
    'eta': 1,
    'nthread': 4,
    'eval_metric': 'auc'
}


# %%
def smape(predt: np.ndarray, dtrain: np.ndarray):
    s = 100 / len(predt) * np.sum(
        2 * np.abs(dtrain - predt) / (np.abs(predt) + np.abs(dtrain))
    )

    return s


def smape_obj(predt: np.ndarray, dtrain: xgb.DMatrix):
    s = smape(
        predt,
        dtrain.get_label()
    )

    return 'smape', s


# %%
num_round = 10
model = xgb.train(param, dtrain, num_round)

# %%
ypred = model.predict(dtest)

# %%
smape(y_test, ypred)

# %%
xgb.plot_importance(model)

# %%
num_round = 10
results = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # clean data
    X_train = pd.get_dummies(X_train, columns=['country', 'store', 'product'])
    X_test = pd.get_dummies(X_test, columns=['country', 'store', 'product'])
    X_train['date'] = pd.to_datetime(X_train['date']).astype('int64') // 10 ** 9
    X_test['date'] = pd.to_datetime(X_test['date']).astype('int64') // 10 ** 9

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    m = xgb.train(param, dtrain, num_round)
    y_pred = m.predict(dtest)
    results.append(smape(y_test, y_pred))

# np.array(results).mean()
results

# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
