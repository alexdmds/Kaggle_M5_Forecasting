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
# Analyse de la date de lancement et d'arrêt des produits
# Pour chaque produit, on cherche le premier et le dernier jour où il a été vendu (vente > 0)
d_cols = [col for col in sales_val.columns if col.startswith('d_')]
launch_days = []
end_days = []
for idx, row in sales_val.iterrows():
    sales = row[d_cols].values
    nonzero = np.where(sales > 0)[0]
    if len(nonzero) > 0:
        launch_days.append(nonzero[0])
        end_days.append(nonzero[-1])
    else:
        launch_days.append(np.nan)
        end_days.append(np.nan)
sales_val['launch_day_idx'] = launch_days
sales_val['end_day_idx'] = end_days

# Conversion en date réelle via calendar
day_map = {f'd_{i+1}': date for i, date in enumerate(calendar['date'])}
launch_dates = [day_map[d_cols[idx]] if not np.isnan(idx) else None for idx in sales_val['launch_day_idx']]
end_dates = [day_map[d_cols[idx]] if not np.isnan(idx) else None for idx in sales_val['end_day_idx']]
sales_val['launch_date'] = launch_dates
sales_val['end_date'] = end_dates

# %%
# Visualisation de la distribution des dates de lancement et d'arrêt
plt.figure(figsize=(12, 4))
sns.histplot(pd.to_datetime(sales_val['launch_date']), bins=50, kde=True)
plt.title('Distribution des dates de lancement des produits')
plt.xlabel('Date de première vente')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
sns.histplot(pd.to_datetime(sales_val['end_date']), bins=50, kde=True)
plt.title('Distribution des dates de dernière vente des produits')
plt.xlabel('Date de dernière vente')
plt.tight_layout()
plt.show()

# %%
# Exemples de produits lancés tardivement ou arrêtés avant la fin
print('Produits lancés après le début de la fenêtre temporelle :')
late_launch = sales_val[sales_val['launch_day_idx'] > 0]
print(late_launch[['id', 'launch_date', 'end_date']].head())

print('\nProduits arrêtés avant la fin de la fenêtre temporelle :')
early_end = sales_val[sales_val['end_day_idx'] < len(d_cols)-1]
print(early_end[['id', 'launch_date', 'end_date']].head())

# %%
# Proportion de produits lancés après le début ou arrêtés avant la fin
prop_late = (sales_val['launch_day_idx'] > 0).mean()
prop_early_end = (sales_val['end_day_idx'] < len(d_cols)-1).mean()
print(f"{prop_late*100:.2f}% des produits ont été lancés après le début de la fenêtre.")
print(f"{prop_early_end*100:.2f}% des produits ont été arrêtés avant la fin de la fenêtre.")

# %%
# Analyse de la couverture des prix dans sell_prices.csv
# Vérifions s'il existe un prix pour chaque (item_id, store_id, semaine) présent dans les ventes
ventes_items = sales_val[['item_id', 'store_id']].drop_duplicates()
ventes_semaines = calendar['wm_yr_wk'].unique()
prix_items = sell_prices[['item_id', 'store_id', 'wm_yr_wk']].drop_duplicates()

# Nombre total de combinaisons possibles (item, store, semaine)
total_combi = len(ventes_items) * len(ventes_semaines)
prix_combi = len(prix_items)
print(f"Nombre de combinaisons (item, store, semaine) possibles : {total_combi}")
print(f"Nombre de combinaisons présentes dans sell_prices : {prix_combi}")
print(f"Couverture des prix : {prix_combi/total_combi*100:.2f}%")

# %%
# Y a-t-il des semaines sans prix pour des articles vendus ?
# On croise les ventes effectives avec les prix
# On cherche les (item_id, store_id, wm_yr_wk) présents dans les ventes mais absents de sell_prices
# Pour cela, on doit "déplier" les ventes par jour et les relier à la semaine
sales_melt = sales_val.melt(
    id_vars=['item_id', 'store_id'],
    value_vars=[col for col in sales_val.columns if col.startswith('d_')],
    var_name='d', value_name='sales')
sales_melt = sales_melt[sales_melt['sales'] > 0]
sales_melt = sales_melt.merge(calendar[['d', 'wm_yr_wk']], on='d', how='left')
ventes_prix = sales_melt[['item_id', 'store_id', 'wm_yr_wk']].drop_duplicates()
prix_merge = ventes_prix.merge(sell_prices, on=['item_id', 'store_id', 'wm_yr_wk'], how='left', indicator=True)
nb_sans_prix = (prix_merge['_merge'] == 'left_only').sum()
print(f"Nombre de (item, store, semaine) vendus sans prix associé : {nb_sans_prix}")

# %%
# Analyse : si le prix n'existe pas, le produit n'est-il pas encore lancé ?
# On regarde la date de lancement des produits concernés
if nb_sans_prix > 0:
    sans_prix = prix_merge[prix_merge['_merge'] == 'left_only']
    print(sans_prix.head())
    # On peut croiser avec la date de lancement calculée précédemment
    print('Dates de lancement des produits sans prix :')
    print(sales_val[sales_val['item_id'].isin(sans_prix['item_id'])][['item_id', 'launch_date']].drop_duplicates().head())
else:
    print("Tous les produits vendus ont un prix associé à leur semaine de vente.")

# %%
# Les prix sont-ils similaires dans tous les magasins ?
# On prend un exemple d'article vendu dans plusieurs magasins
ex_item = sell_prices['item_id'].value_counts().index[0]
prix_ex = sell_prices[sell_prices['item_id'] == ex_item]
plt.figure(figsize=(10, 5))
sns.lineplot(x='wm_yr_wk', y='sell_price', hue='store_id', data=prix_ex, marker='o')
plt.title(f"Évolution du prix pour l'article {ex_item} selon les magasins")
plt.xlabel('Semaine')
plt.ylabel('Prix')
plt.legend(title='Magasin')
plt.tight_layout()
plt.show()

# %%
# Statistiques de dispersion des prix pour chaque article
prix_disp = sell_prices.groupby('item_id')['sell_price'].agg(['mean', 'std', 'min', 'max', 'nunique'])
prix_disp['cv'] = prix_disp['std'] / prix_disp['mean']
print('Exemples de dispersion des prix par article :')
print(prix_disp.sort_values('cv', ascending=False).head())

plt.figure(figsize=(8, 4))
sns.histplot(prix_disp['cv'].dropna(), bins=50)
plt.title('Distribution du coefficient de variation des prix par article')
plt.xlabel('CV (écart-type / moyenne)')
plt.tight_layout()
plt.show()

# %%
# Analyse de la sparsité des ventes : proportion de zéros
sales_values = sales_val[[col for col in sales_val.columns if col.startswith('d_')]].values.flatten()
prop_zeros = (sales_values == 0).mean()
print(f"Proportion de zéros dans toutes les ventes : {prop_zeros*100:.2f}%")

# Proportion de zéros par produit
zero_by_product = (sales_val[[col for col in sales_val.columns if col.startswith('d_')]] == 0).mean(axis=1)
plt.figure(figsize=(8, 4))
sns.histplot(zero_by_product, bins=50)
plt.title('Distribution de la proportion de zéros par produit')
plt.xlabel('Proportion de jours sans vente')
plt.tight_layout()
plt.show()

# %%
# Visualisation de la distribution des ventes (zéros vs non-zéros)
plt.figure(figsize=(8, 4))
sns.histplot(sales_values[sales_values > 0], bins=50, kde=True)
plt.title('Distribution des ventes (hors zéros)')
plt.xlabel('Ventes (uniquement jours avec vente)')
plt.tight_layout()
plt.show()

# %%
# Exemples de séries temporelles avec pics de ventes
sample_ids = sales_val['id'].sample(5, random_state=42)
plt.figure(figsize=(12, 6))
for id_ in sample_ids:
    d_cols = [col for col in sales_val.columns if col.startswith('d_')]
    plt.plot(range(len(d_cols)), sales_val[sales_val['id'] == id_][d_cols].values.flatten(), label=id_)
plt.title('Exemple de séries temporelles de ventes (pics et zéros)')
plt.xlabel('Jour')
plt.ylabel('Ventes')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Commentaire pédagogique
print("La grande majorité des ventes sont des zéros (pas de vente la plupart des jours), avec des pics occasionnels. Cela reflète une demande sporadique ou des achats groupés lors de promotions ou événements.")

# %%
