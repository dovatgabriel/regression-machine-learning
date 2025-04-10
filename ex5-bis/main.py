import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

print("Aperçu du jeu de données :")
print("Premières lignes :")
print(df.head(), "\n")
print("Dernières lignes :")
print(df.tail(), "\n")

print("Informations de base :")
print(f"Nombre de lignes : {df.shape[0]}")
print(f"Nombre de colonnes : {df.shape[1]}\n")

print("Types de données selon le ML :")
data_types_ml = {
    'Nominale': ['Sex', 'Embarked', 'Survived'],
    'Ordinale': ['Pclass'],
    'Discrète': ['SibSp', 'Parch'],
    'Continue': ['Age', 'Fare']
}
for dtype, columns in data_types_ml.items():
    print(f"- {dtype} : {columns}")
print()

print("Survivants vs non-survivants :")
print(df['Survived'].value_counts().rename(index={0: 'Non-survivants', 1: 'Survivants'}))
print()

print("Distribution des passagers :")
print("Par classe :")
print(df['Pclass'].value_counts(), "\n")

print("Par sexe :")
print(df['Sex'].value_counts(), "\n")

print("Par port d'embarquement :")
print(df['Embarked'].value_counts(), "\n")

print("Distribution de l'âge (histogramme) :")
plt.figure(figsize=(10, 5))
df['Age'].hist(bins=30, edgecolor='black')
plt.title("Distribution de l'âge des passagers")
plt.xlabel("Âge")
plt.ylabel("Nombre de passagers")
plt.grid(False)
plt.tight_layout()
plt.show()
