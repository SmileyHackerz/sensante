import pandas as pd

# ===== CHARGER LES DONNEES =====
df = pd.read_csv("data/patients_dakar.csv")

# ===== PREMIERS APER US =====
print("=" * 50)
print("SENSANTE- Exploration du dataset")
print("=" * 50)

# Dimensions du dataset
print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")

# Apercu des 5 premieres lignes
print(f"\n--- 5 premiers patients---")
print(df.head())

# ===== STATISTIQUES DE BASE =====
print(f"\n--- Statistiques descriptives---")
print(df.describe().round(2))

# ===== REPARTITION DES DIAGNOSTICS =====
print(f"\n--- Repartition des diagnostics---")
diag_counts = df["diagnostic"].value_counts()
for diag, count in diag_counts.items():
    pct = count / len(df) * 100
    print(f" {diag:12s} : {count:3d} patients ({pct:.1f}%)")

# ===== REPARTITION PAR REGION =====
print(f"\n--- Repartition par region (top 5)---")
region_counts = df["region"].value_counts().head(5)
for region, count in region_counts.items():
    print(f" {region:15s} : {count:3d} patients")

# ===== ANALYSE PAR SEXE ET DIAGNOSTIC (EXERCICE 1) =====
print(f"\n--- Nombre de patients par sexe et diagnostic ---")
# Utilisation de groupby pour compter les patients par sexe et par pathologie
analyse_sexe_diag = df.groupby(["sexe", "diagnostic"]).size()
print(analyse_sexe_diag)

# ===== TEMPERATURE MOYENNE PAR DIAGNOSTIC =====
print(f"\n--- Temperature moyenne par diagnostic---")
temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
for diag, temp in temp_by_diag.items():
    print(f" {diag:12s} : {temp:.1f} C")

print(f"\n{'=' * 50}")
print("Exploration terminee !")
print("Prochain lab : entrainer un modele ML")
print(f"{'=' * 50}")

from sklearn.model_selection import train_test_split
# 80% pour l'entrainement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.2,
# 20% pour le test
random_state=42,
# Pour avoir les memes resultats a chaque fois
stratify=y
# Garder les memes proportions de diagnostics
)
print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")