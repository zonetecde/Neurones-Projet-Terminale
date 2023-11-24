#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:01:48 2023

@author: matt
"""

from typing import Dict, Tuple, List
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


    def activation(self, entrees: Tuple[float, float], poids: List[int]):
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
    
    def apprentissage(self, entrees: Tuple[float, float], poids: List[int], objectif: str):
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
        
    
    def entrainement(self, jeu_tests: Dict[int, Tuple[Tuple[float, float], str]]) -> None:
        """
        Entraine le neurone sur un jeu de teste

        Parameters
        ----------
        jeu_tests : dict
            Jeu de teste {id: int, [entrees: list, objectif: int]}.
        """
        etat = [1 for i in range(self.n)]
 
        for value in jeu_tests.values():
            entrees = value[0]
            objectif = value[1]
            etat = self.apprentissage(entrees, etat, objectif)
            
        self.etat = etat
    
    
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
    
    def metrique(self, jeu_tests: Dict[int, Tuple[Tuple[float, float], str]], verbose: bool = True):
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
    
    @staticmethod # Pour ne pas avoir à rappeler cette méthode à chaque itération
    def chargement_salary(nom_fichier: str, comparateur_salaire_voulu: float):
        """ 
        Chargement d'un fichier et créaction du dictionnaire
        :param nom_fichier:_ nom du fichier (str)
        :return: {Id:([age, annee_experience], le_salaire_voulu_sera_depasse} (dict)
        """
        fichier = open(nom_fichier, "r")
        jeu_tests = {}
        fichier.readline()

        counter = 0
        for ligne in fichier:
            ligne = ligne.split(',')
            id_data = counter
            counter += 1
            salaire = float(ligne[2])
            age = float(ligne[1])
            annee_experience = float(ligne[0])

            jeu_tests[id_data] = [(age, annee_experience), salaire > comparateur_salaire_voulu]
        fichier.close()
        return jeu_tests

    @staticmethod # Pour ne pas avoir à rappeler cette méthode à chaque itération 
    def chargement_rouge(nom_fichier: str):
        """ 
        Chargement d'un fichier et créaction du dictionnaire
        :param nom_fichier:_ nom du fichier (str)
        :return: {Id:([r, g, b], est_rouge} (dict)
        """
        fichier = open(nom_fichier, "r")
        jeu_tests = {}
        fichier.readline()

        for ligne in fichier:
            ligne = ligne.split(',')
            id_data = int(ligne[0])
            r = int(ligne[1])
            g = int(ligne[2])
            b = int(ligne[3])
            est_rouge = ligne[4].replace('\n','')

            if est_rouge[0] == '1':
                est_rouge = True
            elif est_rouge[0] == '0':
                est_rouge = False

            jeu_tests[id_data] = [(r, g, b), est_rouge]
        fichier.close()
        return jeu_tests
    
    @staticmethod # Pour ne pas avoir à rappeler cette méthode à chaque itération 
    def chargement_titanic(nom_fichier: str):
        """ 
        Chargement d'un fichier et créaction du dictionnaire
        :param nom_fichier:_ nom du fichier (str)
        :return: {Id:([classe, sexe, age], est_survivant} (dict)
        """
        fichier = open(nom_fichier, "r")
        jeu_tests = {}
        fichier.readline()

        for ligne in fichier:
            ligne = ligne.split(',')
            id_data = int(ligne[0])
            est_survivant = int(ligne[1])
            classe = int(ligne[2])
            sexe = 0 if ligne[4] == "male" else 1 # Homme = 0, femme = 1

            if(ligne[5] == ''):
                continue # passe cette ligne car l'âge n'est pas renseigné
            age = float(ligne[5])

            jeu_tests[id_data] = [(classe, sexe, age), True if est_survivant == 1 else False]
        fichier.close()
        return jeu_tests

    def trace(self, jeu_tests: Tuple[float, float]) -> None:
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

    def trouver_meilleur_alpha(self, train_set, test_set, pas: float = 0.001, nbre_n = 2) -> float:
        """Trouve le meilleur alpha avec un pas précisé

        Args:
            pas (float, optional): Le pas. Defaults to 0.001.

        Returns:
            tuple : Index 0 : Le pourcentage de précision
                    Index 1 : Liste des alphas ayant ce pourcentage de précision
        """
        # format : pourcentage de réussite -> liste des alphas ayant ce pourcentage
        resultats: Dict[int, list[float]] = {}

        # utilise np pour pouvoir for loop des nombres à virgule
        for i in np.arange(0, 1, pas):
            alpha =  round(i, 4)
            perceptron = Perceptron(nbre_n, 1, alpha)       
            perceptron.entrainement(train_set)
            pourcentage = perceptron.metrique(test_set, verbose=False)

            # ajout le résultat au dictionnaire
            if pourcentage in resultats:
                resultats[pourcentage].append(alpha)
            else:
                resultats[pourcentage] = []

        best_pourcentage = max(resultats, key=resultats.get)
        return (best_pourcentage, resultats[best_pourcentage])
    
    def changer_alpha(self, nouveau_alpha) -> None:
        """Change l'alpha

        Args:
            nouveau_alpha (float): Le nouvelle alpha
        """
        self.alpha = nouveau_alpha

    def sauvegarder_etat(self, nom_fichier: str) -> None:
        """Sauvegarde l'état du perceptron

        Args:
            nom_fichier (str): Le nom du fichier
        """
        fichier = open(nom_fichier, "w")
        fichier.write(str(self.etat))
        fichier.close()

    def charger_etat(self, nom_fichier: str) -> None:
        """Charge l'état du perceptron depuis un fichier

        Args:
            nom_fichier (str): Le nom du fichier
        """
        fichier = open(nom_fichier, "r")
        self.etat = eval(fichier.readline())
        fichier.close()

# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

print("=============================================================")
print("| Bienvenue dans le système neuronique de Rayane STASZEWSKI |")
print("=============================================================")

while(True):
    print("\n")
    print("Choisissez votre chemin dans ce labyrinthe neuronique...")
    print("1. Rouge ou bleu ?")
    print("2. Rouge ou autre couleur ?")
    print("3. Titanic ?")
    print("4. Salaire après X année ?\n")
    print("0. Quitter la demeure")
    print("\nLe saviez-vous ?... Vous pouvez sauvegarder votre perception à l'aide la méthode .sauvegarder_etat(nom_fichier)")

    choix = int(input("Votre choix, si vous nous le permettez... : "))

    print("\n\n")

    if choix == 1:
        TRAIN_SET = Perceptron.chargement_base("rougebleu_train.csv")
        TEST_SET = Perceptron.chargement_base("rougebleu_test.csv")

        perceptron = Perceptron(2, 1, 1)

        # prend le premier de la liste des meilleurs alphas
        # et récupère le premier élément de la liste
        meilleur_alpha = perceptron.trouver_meilleur_alpha(TRAIN_SET, TEST_SET, nbre_n=2)[1][0] 
        perceptron.changer_alpha(meilleur_alpha)

        perceptron.entrainement(TRAIN_SET)
        perceptron.trace(TEST_SET)
        pourcentage = perceptron.metrique(TEST_SET, verbose=True)

        print("L'alpha choisi est", meilleur_alpha, "avec un taux de précision de", pourcentage, "%")

        time.sleep(3)

    elif choix == 2:
        TRAIN_SET = Perceptron.chargement_rouge("detecterouge_train.csv")
        TEST_SET = Perceptron.chargement_rouge("detecterouge_test.csv")

        perceptron = Perceptron(3, 1, 1)

        # prend le premier de la liste des meilleurs alphas
        # et récupère le premier élément de la liste
        meilleur_alpha = perceptron.trouver_meilleur_alpha(TRAIN_SET, TEST_SET, nbre_n=3)[1][0] 
        perceptron.changer_alpha(meilleur_alpha)

        perceptron.entrainement(TRAIN_SET)
        pourcentage = perceptron.metrique(TEST_SET)

        print("L'alpha choisi est", meilleur_alpha, "avec un taux de précision de", pourcentage, "%")
        time.sleep(3)

    elif choix == 3:
        TRAIN_SET = Perceptron.chargement_titanic("titanic_train.csv")
        TEST_SET = Perceptron.chargement_titanic("titanic_test.csv")

        perceptron = Perceptron(3, 1, 1)

        # prend le premier de la liste des meilleurs alphas
        # et récupère le premier élément de la liste
        meilleur_alpha = perceptron.trouver_meilleur_alpha(TRAIN_SET, TEST_SET, nbre_n=3)[1][0] 
        perceptron.changer_alpha(meilleur_alpha)

        perceptron.entrainement(TRAIN_SET)
        pourcentage = perceptron.metrique(TEST_SET)

        print("L'alpha choisi est", meilleur_alpha, "avec un taux de précision de", pourcentage, "%")

        print("Et vous, auriez-vous survécu ?")
        age = int(input("Quel est votre âge ? "))
        sexe = int(input("Quel est votre sexe ? (0 pour homme, 1 pour femme) "))
        classe = int(input("Quelle aurait été votre classe ? "))

        print("Vous auriez", "survécu" if perceptron.reponse((classe, sexe, age)) == 1 else "péri")

        time.sleep(3)

    elif choix == 4:

        print("Entrez votre salaire voulu, après X années d'expérience...")
        salaire_voulu = float(input("Quel est votre salaire voulu ? "))
        annee_experience = float(input("Dans combien d'année d'expérience vous voudriez ce salaire ? (float accepté, ex: 3.5) : "))

        TRAIN_SET = Perceptron.chargement_salary("Salary_Data.csv", salaire_voulu)
        perceptron = Perceptron(2, 1, 1)
        perceptron.entrainement(TRAIN_SET)

        print("Vous atteindrez votre salaire voulu, d'après des données officiels ! " 
              if perceptron.reponse((annee_experience, salaire_voulu)) == 1 
              else "Vous n'atteindrez pas votre salaire voulu, misère...")

        time.sleep(3)


    elif choix == 0:
        print("Au revoir...")
        exit()

