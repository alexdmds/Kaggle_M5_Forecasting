# %%
# Imports
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Téléchargement des données (si non présentes)
if not os.path.exists('sales_train_validation.csv'):
    os.system('kaggle competitions download -c m5-forecasting-accuracy')
    os.system('unzip -n m5-forecasting-accuracy.zip')

# %%
# Chargement et filtrage des données (ex : CA uniquement)
sales = pd.read_csv('sales_train_validation.csv')
sales_ca = sales[sales['store_id'].str.startswith('CA_')].copy()

# %%
# Transformation du format wide vers long
d_cols = [col for col in sales_ca.columns if col.startswith('d_')]
df_long = sales_ca.melt(
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    value_vars=d_cols,
    var_name='d',
    value_name='sales'
)

# %%
# Ajout des données calendaires
calendar = pd.read_csv('calendar.csv')
df_long = df_long.merge(calendar, on='d', how='left')

# %%
# Visualisation des données
df_long.head()
len(df_long)

# %%
# Feature engineering (lags, moyennes mobiles, date parts)
def create_features(df):
    # Lags
    for lag in [7, 28]:
        df[f'lag_{lag}'] = df.groupby('id')['sales'].shift(lag)
    # Moyennes mobiles
    for window in [7, 28]:
        df[f'rolling_mean_{window}'] = df.groupby('id')['sales'].shift(28).rolling(window).mean().reset_index(0, drop=True)
    # Date parts
    df['wday'] = df['wday']
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['day'] = pd.to_datetime(df['date']).dt.day
    return df

df_long = create_features(df_long)

# %%
# Encodage des features catégorielles
cat_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'wday', 'month', 'year']
for col in cat_features:
    le = LabelEncoder()
    df_long[col] = le.fit_transform(df_long[col].astype(str))

# %%
# Split train / validation (derniers 28 jours)
max_d = df_long['d'].apply(lambda x: int(x.split('_')[1])).max()
valid_idx = df_long['d'].apply(lambda x: int(x.split('_')[1]) > max_d - 28)
train_idx = ~valid_idx

features = [
    'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
    'wday', 'month', 'year', 'day',
    'lag_7', 'lag_28', 'rolling_mean_7', 'rolling_mean_28'
]
X_train = df_long.loc[train_idx, features]
y_train = df_long.loc[train_idx, 'sales']
X_valid = df_long.loc[valid_idx, features]
y_valid = df_long.loc[valid_idx, 'sales']

# %%
# Entraînement d'un modèle LightGBM avec objective='poisson'
lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
params = {
    'objective': 'poisson',
    'metric': 'rmse',
    'force_row_wise': True,
    'learning_rate': 0.05,
    'sub_row': 0.75,
    'bagging_freq': 1,
    'lambda_l2': 0.1,
    'verbosity': -1,
    'seed': 42
}
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(100)]
)

# %%
# Évaluation avec RMSE
preds = model.predict(X_valid, num_iteration=model.best_iteration)
rmse = np.sqrt(mean_squared_error(y_valid, preds))
print(f'RMSE validation : {rmse:.4f}')

# %%
# Affichage des importances et d'une courbe de prédiction
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=15)
plt.title('Importance des variables')
plt.show()

# Prédiction sur un produit exemple
example_id = df_long['id'].unique()[0]
example = df_long[(df_long['id'] == example_id) & valid_idx]
example_pred = model.predict(example[features], num_iteration=model.best_iteration)
plt.figure(figsize=(12, 5))
plt.plot(example['date'], example['sales'], label='Vrai')
plt.plot(example['date'], example_pred, label='Prévu')
plt.title(f'Prévision sur le produit {example_id}')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show() 
# %%
