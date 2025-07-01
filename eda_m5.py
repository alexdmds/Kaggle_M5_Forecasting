# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# %%
# Chargement des fichiers CSV
calendar = pd.read_csv('calendar.csv')
sales_val = pd.read_csv('sales_train_validation.csv')
sell_prices = pd.read_csv('sell_prices.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# %%
# Aperçu général des fichiers
print('calendar.csv:', calendar.shape)
print('sales_train_validation.csv:', sales_val.shape)
print('sell_prices.csv:', sell_prices.shape)
print('sample_submission.csv:', sample_submission.shape)

# %%
# Affichage des premières lignes de chaque fichier
print('--- calendar.csv ---')
print(calendar.head())
print('\n--- sales_train_validation.csv ---')
print(sales_val.head())
print('\n--- sell_prices.csv ---')
print(sell_prices.head())
print('\n--- sample_submission.csv ---')
print(sample_submission.head())

# %%
# Types de données et valeurs manquantes
print('--- Types et valeurs manquantes : calendar.csv ---')
print(calendar.info())
print(calendar.isnull().sum())
print('\n--- Types et valeurs manquantes : sales_train_validation.csv ---')
print(sales_val.info())
print(sales_val.isnull().sum())
print('\n--- Types et valeurs manquantes : sell_prices.csv ---')
print(sell_prices.info())
print(sell_prices.isnull().sum())

# %%
# Statistiques descriptives
print('--- Statistiques descriptives : sales_train_validation.csv ---')
print(sales_val.describe())
print('\n--- Statistiques descriptives : sell_prices.csv ---')
print(sell_prices.describe())

# %%
# Exploration des colonnes catégorielles
print('--- Colonnes catégorielles : sales_train_validation.csv ---')
for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
    print(f'{col}: {sales_val[col].nunique()} valeurs uniques')
    print(sales_val[col].value_counts().head())
    print()

print('--- Colonnes catégorielles : calendar.csv ---')
for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
    print(f'{col}: {calendar[col].nunique()} valeurs uniques')
    print(calendar[col].value_counts(dropna=False))
    print()

print('--- Colonnes catégorielles : sell_prices.csv ---')
for col in ['store_id', 'item_id']:
    print(f'{col}: {sell_prices[col].nunique()} valeurs uniques')
    print(sell_prices[col].value_counts().head())
    print()

# %%
# Distribution des ventes (exemple sur un échantillon)
sample_ids = sales_val['id'].sample(5, random_state=42)
plt.figure(figsize=(12, 6))
for i, id_ in enumerate(sample_ids):
    d_cols = [col for col in sales_val.columns if col.startswith('d_')]
    plt.plot(range(len(d_cols)), sales_val[sales_val['id'] == id_][d_cols].values.flatten(), label=id_)
plt.title('Exemple de séries temporelles de ventes (5 produits)')
plt.xlabel('Jour')
plt.ylabel('Ventes')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Distribution globale des ventes (tous produits confondus)
all_sales = sales_val[[col for col in sales_val.columns if col.startswith('d_')]].values.flatten()
plt.figure(figsize=(8, 4))
sns.histplot(all_sales, bins=50, kde=True)
plt.title('Distribution globale des ventes (tous produits, tous jours)')
plt.xlabel('Ventes')
plt.ylabel('Fréquence')
plt.tight_layout()
plt.show()

# %%
# Analyse des prix
print('--- Statistiques des prix par magasin ---')
print(sell_prices.groupby('store_id')['sell_price'].describe())

plt.figure(figsize=(10, 5))
sns.boxplot(x='store_id', y='sell_price', data=sell_prices)
plt.title('Distribution des prix par magasin')
plt.tight_layout()
plt.show()

# %%
# Analyse calendaire : nombre d'événements par type
for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
    plt.figure(figsize=(8, 3))
    calendar[col].value_counts(dropna=False).plot(kind='bar')
    plt.title(f'Distribution de {col}')
    plt.tight_layout()
    plt.show()

# %%
# Corrélation entre prix et ventes (exemple sur un produit)
example_item = sell_prices['item_id'].unique()[0]
example_store = sell_prices['store_id'].unique()[0]
prices = sell_prices[(sell_prices['item_id'] == example_item) & (sell_prices['store_id'] == example_store)]

sales_row = sales_val[(sales_val['item_id'] == example_item) & (sales_val['store_id'] == example_store)]
if not sales_row.empty:
    sales_series = sales_row[[col for col in sales_row.columns if col.startswith('d_')]].T.reset_index(drop=True)
    sales_series.columns = ['sales']
    sales_series['wm_yr_wk'] = calendar['wm_yr_wk'][:len(sales_series)]
    merged = sales_series.merge(prices, on='wm_yr_wk', how='left')
    plt.figure(figsize=(10, 4))
    plt.plot(merged['sell_price'], label='Prix')
    plt.plot(merged['sales'], label='Ventes')
    plt.title(f'Prix vs Ventes pour {example_item} ({example_store})')
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
# Heatmap de corrélation sur les features numériques de sell_prices
plt.figure(figsize=(6, 4))
sns.heatmap(sell_prices.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Corrélation des variables numériques (sell_prices)')
plt.tight_layout()
plt.show()

# %%
# Aperçu du format de soumission
print('--- Format de soumission ---')
print(sample_submission.head())
print('Colonnes:', sample_submission.columns.tolist()) 
# %%
