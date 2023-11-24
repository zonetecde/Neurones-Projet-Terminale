import csv
import random

# Fonction pour générer des données aléatoires

def generate_random_data(id = 0):
    id += 1
    age = random.randint(16, 18)
    sexe = random.choice([0, 1])
    classe = random.choice([1, 2, 3])
    moyenne_nsi = round(random.uniform(0, 20), 2)
    return [id, age, sexe, classe, moyenne_nsi]

# Création du fichier CSV
with open('donnees.csv', 'w', newline='') as csvfile:
    fieldnames = ['Age', 'Sexe', 'Classe', 'Moyenne NSI']
    writer = csv.writer(csvfile)
    
    # Écriture de l'en-tête
    writer.writerow(fieldnames)
    
    # Génération de 100 données aléatoires
    for _ in range(1000):
        data = generate_random_data(_)
        writer.writerow(data)

print("Le fichier CSV a été créé avec succès.")