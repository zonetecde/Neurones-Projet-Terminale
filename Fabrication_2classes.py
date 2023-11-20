
import pandas as pd
import random as rd
import numpy as np
import matplotlib.pyplot as plt


def affine(a, b, x):
    """
    Calcul de l'ordonnée y connaisant l'abscisse x
    :param a: coefficient directeur de la droite  (int)
    :param b: ordonnée à l'origine (int)
    :return: l'abscisse y (int)
    """

    y = a*x + b
    # print(int(y))

    return int(y)


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================


# Spécification de la droite (pente,ordonnée à l'origine) -5x +50
a, b = -5, 50
# valeur maximale de x et y
x_min, x_max = -50, 50

# Calcul de ymin et ymax
y_min, y_max = affine(a, b, x_min), affine(a, b, x_max)
if y_min > y_max:
    y_min, y_max = y_max, y_min

# Liste contenant les tuples (x,y,couleur)
l_red, l_blu = [], []


# Fabrication des points
NB_POINTS = 1000
for i in range(0, NB_POINTS//2):
    # Choix de 2 y au hasard
    x_red, x_blu = rd.randint(x_min, x_max), rd.randint(x_min, x_max)
    # Calcul du y correspondant de part et d'autre de la droite
    y_red = rd.randint(y_min, affine(a, b, x_red))
    y_blu = rd.randint(affine(a, b, x_blu), y_max)
    l_red.append((x_red, y_red, 'r'))
    l_blu.append((x_blu, y_blu, 'b'))


# Liste permettant le tracé des points
lxr, lyr = [], []
lxb, lyb = [], []

for i in range(0, len(l_red)):
    lxr.append(l_red[i][0])
    lyr.append(l_red[i][1])
    lxb.append(l_blu[i][0])
    lyb.append(l_blu[i][1])


# Tracé des points
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.scatter(lxr, lyr, c='red', marker='o')
plt.scatter(lxb, lyb, c='blue', marker='s')
plt.title("Classification")
plt.grid()

# Calcul des coordonnées de la droite
x_droite = np.arange(x_min, x_max)
y_droite = a*x_droite+b

plt.plot(x_droite, y_droite, 'g')
plt.show()

# Fabrication et export du dataframe en csv
data = l_red+l_blu
rd.shuffle(data)
# print(data)

# Séparation du train set et du test set
POURCENTAGE = 0.8

limite = int(len(data)*POURCENTAGE)

df_train = pd.DataFrame(columns=['x', 'y', 'coul'], data=data[:limite])
df_test = pd.DataFrame(columns=['x', 'y', 'coul'], data=data[limite:])

df_train.to_csv("rougebleu_train.csv")
df_test.to_csv("rougebleu_test.csv")

