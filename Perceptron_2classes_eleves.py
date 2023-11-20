#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:01:48 2023

@author: matt
"""

from typing import Dict
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import time

class Perceptron:
    """ 
    Création de l'objet Perceptron 
    """
    
    def __init__(self, n:int, biais:float, alpha:float):
        """
        Initialisation d'objet Perceptron
        Params:
            n: int: nbre d'entrée
            biais: float
            alpha: float: taux d'apprentissage
        """
        self.n = n
        self.biais = biais
        self.alpha = alpha
        
        self.etat = [1 for i in range(n)]


    def activation(self, entrees: tuple[float, float], poids: list[int]):
        """
        Activeur
        
        params:
            entrees (liste) : Les entrées
            poids (liste) : Les poids
            
        Return: 1 en cas d'activation, 0 sinon
        """
        results = 0
        
        assert len(entrees) == len(poids)
        
        for i in range(len(entrees)):
            results += entrees[i] * poids[i]
        
        return int(results >= self.biais)
    
    def apprentissage(self, entrees: tuple[float, float], poids: list[int], objectif: str):
        """
        Renvoie un nouvel état modifié du perceptron

        Parameters
        ----------
        entrees : list
            Les entrées.
        poids : list
            Les poids.
        objectif : float
            Objectif voulu.

        Returns: list:  Nouvel état
        """
        nouveaux_poids = []
        
        # Récupère l'activeur
        valeur = self.activation(entrees, poids)
        
        # Trouve le signe 
        signe = 0
        
        if valeur == 0 and objectif == 1:
            signe = 1
        elif valeur == 1 and objectif == 0:
            signe = -1
        
        for i in range(len(entrees)):
            # Calcul du nouveau poid
            nouveaux_poids.append(poids[i] + signe * self.alpha * entrees[i])
            
        return nouveaux_poids
        
    
    def entrainement(self, jeu_tests: Dict[int, list[tuple[float, float], str]]):
        """
        Entraine le neurone sur un jeu de teste

        Parameters
        ----------
        jeu_tests : dict
            Jeu de teste {id: int, [entrees: list, objectif: int]}.

        Returns : Etat final
        """
        etat = [1 for i in range(self.n)]
 
        for (id, value) in jeu_tests.items():
            entrees = value[0]
            objectif = value[1]
            etat = self.apprentissage(entrees, etat, objectif)
            
        self.etat = etat
        return etat
    
    
    def reponse(self, entrees):
        """
        Renvoie une réponse en fonction des entrées entrainées

        Parameters
        ----------
        entrees : list
            Les entrées entrainées.

        Returns
        -------
        int
            1 si positif, 0 sinon.

        """
        return  self.activation(entrees, self.etat)
    
    def metrique(self, jeu_tests: Dict[int, list[tuple[float, float], str]], verbose: bool = True):
        """Calcul le nombre de prédiction correct en pourcentage

        Args:
            jeu_tests (dict): Les jeux de testes

        Returns:
            float: Pourcentage de prédiction correct
        """
        correct_predictions = 0
        
        for jeu in jeu_tests.values():
            # jeu est au format [entrees, objectif]
            prediction = self.reponse(jeu[0])
                
            if prediction == jeu[1]:
                correct_predictions += 1

        if verbose:
            print("Nbre d'erreur(s) : ", len(jeu_tests) - correct_predictions)

        return round(correct_predictions / len(jeu_tests) * 100)
        

    @staticmethod # Pour ne pas avoir à rappeler cette méthode à chaque itération
    def chargement_base(nom_fichier: str):
        """ 
        Chargement d'un fichier et créaction du dictionnaire
        :param nom_fichier:_ nom du fichier (str)
        :return: {Id:([Abscisse, ordonnée], couleur} (dict)
        """
        fichier = open(nom_fichier, "r")
        jeu_tests = {}
        fichier.readline()

        for ligne in fichier:
            ligne = ligne.split(',')
            id_data = int(ligne[0])
            x = int(ligne[1])
            y = int(ligne[2])
            couleur = ligne[3]

            if couleur[0] == 'r':
                couleur = 0
            elif couleur[0] == 'b':
                couleur = 1

            jeu_tests[id_data] = [(x, y), couleur]
        fichier.close()
        return jeu_tests


    def trace(self, jeu_tests: tuple[float, float]):
        """ 
        Affichage des features et de la frontière de décision
        :param jeu_tests: jeu de tests (dict)
        :return: None
        """
        # Création des coordonnées pour les rouges et les bleus
        lxr, lyr = [], []
        lxb, lyb = [], []
        for val in jeu_tests.values():
            if val[1] == 0:
                lxr.append(val[0][0])
                lyr.append(val[0][1])
            else:
                lxb.append(val[0][0])
                lyb.append(val[0][1])
        
        x_min, x_max = min(min(lxr),min(lxb)), max(max(lxr),max(lxb))
        y_min, y_max = min(min(lyr),min(lyb)), max(max(lyr),max(lyb))
        
        # Calcul du coefficient directeur et de l'ordonnée à l'origine
        w1, w2 = self.etat
        a = -w1/w2
        b = self.biais/w2
        
        # Tracé des points
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.scatter(lxr, lyr, c='red', marker='o')
        plt.scatter(lxb, lyb, c='blue', marker='s')
        plt.title("Classification")
        plt.grid()
        x_droite = np.arange(x_min, x_max, 10)
        y_droite = a*x_droite+b
        plt.plot(x_droite, y_droite, 'g')
        plt.show()


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

TRAIN_SET = Perceptron.chargement_base("rougebleu_train.csv")
TEST_SET = Perceptron.chargement_base("rougebleu_test.csv")

# format : pourcentage de réussite -> liste des alphas ayant ce pourcentage
resultats: Dict[int, list[float]] = {}

# utilise np pour pouvoir for loop des nombres à virgule
for i in np.arange(0, 1, 0.001):
    alpha =  round(i, 4)
    perceptron = Perceptron(2, 1, alpha)       
    resultat = perceptron.entrainement(TRAIN_SET)
    #perceptron.trace(perceptron.chargement_base("rougebleu_test.csv"))
    pourcentage = perceptron.metrique(TEST_SET, verbose=False)

    # ajout le résultat au dictionnaire
    if pourcentage in resultats:
        resultats[pourcentage].append(alpha)
    else:
        resultats[pourcentage] = []

best_pourcentage = max(resultats, key=resultats.get)

print("Le(s) meilleur(s) alpha sont/est", resultats[best_pourcentage], "avec une précision de", best_pourcentage, "%")

perceptron = Perceptron(2, 1, resultats[best_pourcentage][0])
resultat = perceptron.entrainement(TRAIN_SET)
perceptron.trace(TEST_SET)
pourcentage = perceptron.metrique(TEST_SET)

print((pourcentage, i))
