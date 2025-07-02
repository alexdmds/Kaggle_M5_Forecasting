import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import gc
import time
import sys

# -----------------------------
# Paramètres de debug
DEBUG_MODE = True  # Passe à False pour le full dataset
DEBUG_N_ITEMS = 8000
DEBUG_N_DAYS = 400

# -----------------------------
# 1. Chargement et optimisation mémoire
# -----------------------------
def reduce_mem_usage(df):
    """Réduit l'utilisation mémoire d'un DataFrame (ne traite que les colonnes numériques)"""
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)[:3] == 'int' or str(col_type)[:5] == 'float':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= 0:
                    if c_max < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)
        # On ignore les colonnes object/category ici
    return df

# -----------------------------
# 2. Feature engineering
# -----------------------------
def create_features(df):
    print("[INFO] Création des lags...")
    for lag in [1, 7, 14, 28, 56]:
        df[f'lag_{lag}'] = df.groupby('id')['sales'].shift(lag)
    print("[INFO] Lags créés.")
    # Indicateurs de zéros
    df['lag_1_zero'] = (df['lag_1'] == 0).astype(np.uint8)
    df['lag_7_zero'] = (df['lag_7'] == 0).astype(np.uint8)
    print("[INFO] Indicateurs de zéros créés.")
    # Moyennes mobiles et rolling zero rate
    print("[INFO] Rolling features...")
    for window in [7, 28]:
        df[f'rolling_mean_{window}'] = df.groupby('id')['sales'].shift(28).rolling(window).mean().reset_index(0, drop=True)
        df[f'rolling_std_{window}'] = df.groupby('id')['sales'].shift(28).rolling(window).std().reset_index(0, drop=True)
        df[f'rolling_zero_rate_{window}'] = df.groupby('id')['sales'].shift(28).rolling(window).apply(lambda x: (x==0).mean(), raw=True).reset_index(0, drop=True)
    print("[INFO] Rolling features créées.")
    # Streaks de zéros (corrigé avec transform pour aligner l'index)
    print("[INFO] Calcul des streaks de zéros...")
    df['days_since_last_sale'] = df.groupby('id')['sales'].transform(lambda x: x[::-1].cumsum().where(x[::-1]!=0).ffill().fillna(0)[::-1])
    print("[INFO] Streaks de zéros créés.")
    # Features calendaires
    print("[INFO] Features calendaires...")
    df['wday'] = df['wday'].astype('category')
    df['month'] = pd.to_datetime(df['date']).dt.month.astype('int8')
    df['year'] = pd.to_datetime(df['date']).dt.year.astype('int16')
    df['day'] = pd.to_datetime(df['date']).dt.day.astype('int8')
    print("[INFO] Features calendaires créées.")
    # Prix
    print("[INFO] Features prix...")
    df['price_change_1'] = df.groupby(['id'])['sell_price'].diff(1)
    df['price_change_7'] = df.groupby(['id'])['sell_price'].diff(7)
    df['rolling_mean_price_7'] = df.groupby('id')['sell_price'].rolling(7).mean().reset_index(0, drop=True)
    df['rolling_std_price_28'] = df.groupby('id')['sell_price'].rolling(28).std().reset_index(0, drop=True)
    print("[INFO] Features prix créées.")
    return df

# -----------------------------
# 3. Pipeline principal
# -----------------------------
def main():
    start_time = time.time()
    print("[INFO] Début du pipeline M5 LightGBM")
    # Chargement
    print("[INFO] Chargement des fichiers CSV...")
    sales = pd.read_csv('sales_train_validation.csv')
    calendar = pd.read_csv('calendar.csv')
    prices = pd.read_csv('sell_prices.csv')
    print("[INFO] Réduction mémoire...")
    sales = reduce_mem_usage(sales)
    calendar = reduce_mem_usage(calendar)
    prices = reduce_mem_usage(prices)
    print(f"[INFO] Fichiers chargés en {time.time()-start_time:.1f}s")

    # Mode debug : sous-échantillonnage
    if DEBUG_MODE:
        print(f"[DEBUG] Mode debug activé : {DEBUG_N_ITEMS} produits, {DEBUG_N_DAYS} jours")
        keep_ids = sales['id'].unique()[:DEBUG_N_ITEMS]
        sales = sales[sales['id'].isin(keep_ids)]
        d_cols = [c for c in sales.columns if c.startswith('d_')]
        keep_days = sorted(d_cols, key=lambda x: int(x.split('_')[1]))[-DEBUG_N_DAYS:]
        sales = sales[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] + keep_days]
        calendar = calendar[calendar['d'].isin(keep_days)]
    
    # Transformation wide -> long
    print("[INFO] Transformation wide -> long...")
    d_cols = [c for c in sales.columns if c.startswith('d_')]
    df = sales.melt(id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                    value_vars=d_cols, var_name='d', value_name='sales')
    df = reduce_mem_usage(df)
    print(f"[INFO] Format long obtenu en {time.time()-start_time:.1f}s")
    print(f"[INFO] Mémoire après melt : {df.memory_usage(deep=True).sum() / 1024**2:.2f} Mo")

    # Merge calendar
    print("[INFO] Fusion avec calendar...")
    df = df.merge(calendar, on='d', how='left')
    # Merge prices
    print("[INFO] Fusion avec sell_prices...")
    df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    print(f"[INFO] Mémoire après merge : {df.memory_usage(deep=True).sum() / 1024**2:.2f} Mo")
    # Libération mémoire des fichiers d'origine
    del sales, calendar, prices
    gc.collect()

    # Encodage catégoriel
    print("[INFO] Encodage des variables catégorielles...")
    for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
        df[col] = df[col].astype('category')

    # Feature engineering
    print("[INFO] Feature engineering...")
    t0 = time.time()
    df = create_features(df)
    print(f"[INFO] Features créées en {time.time()-t0:.1f}s")
    print(f"[INFO] Mémoire après features : {df.memory_usage(deep=True).sum() / 1024**2:.2f} Mo")

    # Ajout de la colonne horizon (allégé : horizons 1, 7, 14, 28)
    print("[INFO] Génération du dataset empilé (horizons 1, 7, 14, 28)...")
    dfs = []
    for h in [1, 7, 14, 28]:
        tmp = df.copy()
        tmp['horizon'] = h
        tmp['target'] = tmp.groupby('id')['sales'].shift(-h)
        dfs.append(tmp)
        del tmp
        gc.collect()
    data = pd.concat(dfs)
    del dfs, df
    gc.collect()
    print(f"[INFO] Dataset empilé prêt en {time.time()-start_time:.1f}s")
    print(f"[INFO] Mémoire après concat : {data.memory_usage(deep=True).sum() / 1024**2:.2f} Mo")

    # Sauvegarde parquet
    data.to_parquet("train_features_m5.parquet")
    print("[INFO] Dataset d'entraînement sauvegardé en train_features_m5.parquet")

    # Filtrage : on garde la dernière année, on retire les lignes avec NaN dans les lags/rolling/target
    print("[INFO] Filtrage des données...")
    min_d = int(data['d'].str.split('_').str[1].astype(int).max()) - 365
    data = data[data['d'].str.split('_').str[1].astype(int) > min_d]
    data = data.dropna(subset=['lag_28', 'rolling_mean_7', 'target'])
    print(f"[INFO] Données filtrées en {time.time()-start_time:.1f}s")
    print(f"[INFO] Mémoire après filtrage : {data.memory_usage(deep=True).sum() / 1024**2:.2f} Mo")

    # Split train/validation (val = d_1886 à d_1913)
    print("[INFO] Split train/validation...")
    max_d = data['d'].str.split('_').str[1].astype(int).max()
    val_mask = data['d'].str.split('_').str[1].astype(int).between(max_d-27, max_d)
    train_mask = ~val_mask

    features = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
        'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
        'wday', 'month', 'year', 'day',
        'sell_price', 'price_change_1', 'price_change_7', 'rolling_mean_price_7', 'rolling_std_price_28',
        'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56',
        'lag_1_zero', 'lag_7_zero',
        'rolling_mean_7', 'rolling_mean_28', 'rolling_std_7', 'rolling_std_28',
        'rolling_zero_rate_7', 'rolling_zero_rate_28',
        'days_since_last_sale',
        'horizon'
    ]

    X_train = data.loc[train_mask, features]
    y_train = data.loc[train_mask, 'target']
    X_val = data.loc[val_mask, features]
    y_val = data.loc[val_mask, 'target']
    print(f"[INFO] Split terminé. Taille train: {X_train.shape}, val: {X_val.shape}")

    # Entraînement LightGBM
    print("[INFO] Entraînement LightGBM...")
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature='auto')
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature='auto')
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.2,
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 128,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'seed': 42,
        'verbosity': -1
    }
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    print(f"[INFO] Entraînement terminé en {time.time()-start_time:.1f}s")

    # Évaluation
    print("[INFO] Évaluation sur la validation...")
    preds = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f'[RESULT] RMSE validation : {rmse:.4f}')

    # Importance des features
    print("[INFO] Affichage de l'importance des variables...")
    lgb.plot_importance(model, max_num_features=20)
    print(f"[INFO] Pipeline terminé en {time.time()-start_time:.1f}s")
    print(f"[INFO] Mémoire finale : {X_train.memory_usage(deep=True).sum() / 1024**2:.2f} Mo (train), {X_val.memory_usage(deep=True).sum() / 1024**2:.2f} Mo (val)")

if __name__ == '__main__':
    main() 