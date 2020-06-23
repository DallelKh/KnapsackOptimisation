from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage

import pandas as pd
import numpy as np
from time import perf_counter





####################################################### class B&B avec elagage 
class Sac_a_dos_app:
    """
    Arbre arbre: arbre de B&B
    """
    #poids=0 :poids du sac à dos
    #borneInf=0
    #DataFrame donnees : objets
    #int decision[] :nombre d'exemplaires pour chaque objets
    def __init__(self,poids,csv_file_path):
        Sac_a_dos_app.poids=poids
        
        data=pd.read_csv(csv_file_path)
        utilites=data["gains"]/data["volumes"]
        data["utilites"]=utilites
        data=data.sort_values(by='utilites', ascending=False) #trier les objets par utilité
        Sac_a_dos_app.donnees=data
        self.poids_restant=0
        self.arbre=Arbre()
        Sac_a_dos_app.borneInf=0
        Sac_a_dos_app.decision=[]

        
    def lancer(self):
        index_donnees=Sac_a_dos_app.donnees.index.values #nouvel index des objets triés
        #créer les noeuds du premier objet
        num_premier_obj=index_donnees[0]
        nbr_exemplaires_max=int(Sac_a_dos_app.poids/Sac_a_dos_app.donnees.loc[num_premier_obj,'volumes'])
        pere=self.arbre.racine
        for i in range(nbr_exemplaires_max+1):
            node=self.arbre.creerNoeud(num_premier_obj,i,pere)
            if node.typeNoeud==1:
                self.arbre.ajouterNoeudActif(node)
        
        #boucle principale:
        while (len(self.arbre.noeudsActifs)!=0): #noeuds actifs non encore explorés existants
            noeud_pere=self.arbre.retirerNoeudActif()
            if noeud_pere.evaluation>Sac_a_dos_app.borneInf: #test si le noeud est toujours actif
                #générer les noeuds fils:
                index_obj_pere=np.nonzero(index_donnees==noeud_pere.numeroObjet)[0][0]
                num_obj=index_donnees[index_obj_pere+1]
                self.poids_restant=Sac_a_dos_app.poids-self.arbre.cout_branche(noeud_pere)
                nbr_exemplaires_max=int( self.poids_restant/Sac_a_dos_app.donnees.loc[num_obj,'volumes'])
                for i in range(nbr_exemplaires_max+1):
                    node=self.arbre.creerNoeud(num_obj,i,noeud_pere)
                    if node.typeNoeud==1:
                        self.arbre.ajouterNoeudActif(node)
            
            
            
        
class Arbre:
    """
    Node racine
    DataFrame('noeuds','eval') noeudsActifs : DataFrame des noeuds actifs
    """
    def __init__(self):
        self.racine=Node()
        self.noeudsActifs=pd.DataFrame({'noeuds':[],'eval':[]})
    
    def creerNoeud(self,numeroObjet,nbrExemplaires,pere):
        """
        crée le noeud, l'évalue et donne son type
        """
        n=Node(typeNoeud=1,numeroObjet=numeroObjet,nbrExemplaires=nbrExemplaires,pere=pere) #créer le noeud
        n.evaluer() #évaluation du noeud
        pere.fils.append(n) #optionnel: ajout du noeud à la liste des fils du père
        
        #determination du type du noeud
        index_donnees=Sac_a_dos_app.donnees.index.values
        index_obj_act=np.nonzero(index_donnees==n.numeroObjet)[0][0]
        if index_obj_act+1==len(index_donnees): #noeud feuille (=>non actif)
            n.typeNoeud=0
            if n.evaluation>Sac_a_dos_app.borneInf:
                Sac_a_dos_app.borneInf=n.evaluation 
                #mise à jour solution
                self.gener_decision(n)
                Sac_a_dos_app.decision=pd.Series(Sac_a_dos_app.decision,index=index_donnees)
                
        else: #noeud non feuille
            if n.evaluation>Sac_a_dos_app.borneInf: n.typeNoeud=1
            else: n.typeNoeud=0
        return n
    
    def gener_decision(self,node):
        """
        génère le nombre d'exemplaires des objets par ordre de l'utilité 
        jusqu'à l'objet situé dans node
        """
        Sac_a_dos_app.decision=[]
        while node.typeNoeud!=-1:
            Sac_a_dos_app.decision.insert(0,node.nbrExemplaires) #insertion au début (push)
            node=node.pere
            
    
    def cout_branche(self,node):
        """
        calcule le cout (volume) d'une branche à partir de node jusqu'à la racine
        """
        cout=0
        while node.typeNoeud!=-1:
            volume_obj=Sac_a_dos_app.donnees.loc[node.numeroObjet,'volumes']
            cout+=node.nbrExemplaires*volume_obj
            node=node.pere
        return cout
            
    def ajouterNoeudActif(self,n):
        self.noeudsActifs=self.noeudsActifs.append({'noeuds':n,'eval':n.evaluation},ignore_index=True)
        self.noeudsActifs=self.noeudsActifs.sort_values(by='eval',ascending=False) #stratégie "noeud avec la plus grande évaluation d'abord"
        self.noeudsActifs.reset_index(drop=True,inplace=True)
        
    def retirerNoeudActif(self):
        n=self.noeudsActifs.loc[0,'noeuds']
        self.noeudsActifs=self.noeudsActifs.drop([0],axis=0)
        self.noeudsActifs.reset_index(drop=True,inplace=True)
        n.typeNoeud=0
        return n
        
        
class Node:
    """
    typeNoeud typeNoeud {1:actif,0:inactif-elagué,-1:racine}
    int numeroObjet :{O,1,2,...}
    int nbrExemplaires 
    int evaluation 
    Node pere
    Node[] fils
    """
    def __init__(self,typeNoeud=-1,numeroObjet=None,nbrExemplaires=None,pere=None):
        self.typeNoeud=typeNoeud
        self.numeroObjet=numeroObjet
        self.nbrExemplaires=nbrExemplaires
        self.pere=pere
        self.fils=[]
        
    def evaluer(self): 
        "meme fonctions que celle vue en cours"
        gain_obj=Sac_a_dos_app.donnees.loc[self.numeroObjet,'gains']
        volume_obj=Sac_a_dos_app.donnees.loc[self.numeroObjet,'volumes']
        
        gain_noeud_act=gain_obj*self.nbrExemplaires
        volume_noeud_act=volume_obj*self.nbrExemplaires
        
        pere=self.pere
        gain_noeuds_peres=0
        volume_noeuds_peres=0
        
        while pere.typeNoeud!=-1:
            gain_obj_pere=Sac_a_dos_app.donnees.loc[pere.numeroObjet,'gains']
            gain_noeuds_peres+=gain_obj_pere*pere.nbrExemplaires
            
            volume_obj_pere=Sac_a_dos_app.donnees.loc[pere.numeroObjet,'volumes']
            volume_noeuds_peres+=volume_obj_pere*pere.nbrExemplaires
            
            pere=pere.pere
            
        gain_tot=gain_noeud_act+gain_noeuds_peres
        volume_tot=volume_noeud_act+volume_noeuds_peres
        
        index_donnees=Sac_a_dos_app.donnees.index.values
        index_obj_act=np.nonzero(index_donnees==self.numeroObjet)[0][0]
        if index_obj_act+1==len(index_donnees): #noeud feuille
            self.evaluation=gain_tot
           
        else:
            num_prochain_obj=index_donnees[index_obj_act+1]
            gain_prochain_obj=Sac_a_dos_app.donnees.loc[num_prochain_obj,'gains']
            volume_prochain_obj=Sac_a_dos_app.donnees.loc[num_prochain_obj,'volumes']
            self.evaluation=gain_tot+(Sac_a_dos_app.poids-volume_tot)*gain_prochain_obj/volume_prochain_obj
            
####################################################### end class B&B avec elagage 


##############################################################  class B&B sans elagage 

class Sac_a_dos_app2:
    
    
    """
    Arbre2 arbre: arbre de B&B
    """
    cpt= 0
    #poids=0 :poids du sac à dos
    #borneInf=0
    #DataFrame donnees : objets
    #int decision[] :nombre d'exemplaires pour chaque objets
    def __init__(self,poids,csv_file_path):
        Sac_a_dos_app2.poids=poids
        
        data=pd.read_csv(csv_file_path)
        l= data["volumes"]
        utilites=data["gains"]/data["volumes"]
        data["utilites"]=utilites
        data=data.sort_values(by='utilites', ascending=False) #trier les objets par utilité
        
        
        Sac_a_dos_app2.donnees=data
        self.poids_restant=0

        self.arbre=Arbre2()
        Sac_a_dos_app2.borneInf=0
        Sac_a_dos_app2.decision=[]

        
    def lancer(self):
        index_donnees=Sac_a_dos_app2.donnees.index.values #nouvel index des objets triés
        #créer les noeuds du premier objet
        num_premier_obj=index_donnees[0]
        nbr_exemplaires_max=int(Sac_a_dos_app2.poids/Sac_a_dos_app2.donnees.loc[num_premier_obj,'volumes'])
        pere=self.arbre.racine
        for i in range(nbr_exemplaires_max+1):
            node=self.arbre.creerNoeud(num_premier_obj,i,pere)
            if node.typeNoeud==1:
                Sac_a_dos_app2.cpt += 1
                self.arbre.ajouterNoeudActif(node)
                
                
        #boucle principale:
        while (len(self.arbre.noeudsActifs)!=0): #noeuds actifs non encore explorés existants
            
            noeud_pere=self.arbre.retirerNoeudActif()
                #générer les noeuds fils:
            index_obj_pere=np.nonzero(index_donnees==noeud_pere.numeroObjet)[0][0]
            num_obj=index_donnees[index_obj_pere+1]
            self.poids_restant=Sac_a_dos_app2.poids-self.arbre.cout_branche(noeud_pere)
            nbr_exemplaires_max=int(self.poids_restant/Sac_a_dos_app2.donnees.loc[num_obj,'volumes'])
            for i in range(nbr_exemplaires_max+1):
                node=self.arbre.creerNoeud(num_obj,i,noeud_pere)
                if node.typeNoeud == 1:
                    Sac_a_dos_app2.cpt += 1
                    self.arbre.ajouterNoeudActif(node)
             
              
           
            
            
        
class Arbre2:
    """
    Node2 racine
    DataFrame('noeuds','eval') noeudsActifs : DataFrame des noeuds actifs
    """
    def __init__(self):
        self.racine=Node2()
        self.noeudsActifs=pd.DataFrame({'noeuds':[],'eval':[]})
    
    def creerNoeud(self,numeroObjet,nbrExemplaires,pere):
        """
        crée le noeud, l'évalue et donne son type
        """
        n=Node2(typeNoeud=1,numeroObjet=numeroObjet,nbrExemplaires=nbrExemplaires,pere=pere) #créer le noeud
        n.evaluer() #évaluation du noeud
        pere.fils.append(n) #optionnel: ajout du noeud à la liste des fils du père
        
        #determination du type du noeud
        index_donnees=Sac_a_dos_app2.donnees.index.values
        index_obj_act=np.nonzero(index_donnees==n.numeroObjet)[0][0]
        if index_obj_act+1==len(index_donnees): #noeud feuille (=>non actif)
            n.typeNoeud=0
            if n.evaluation>Sac_a_dos_app2.borneInf:
                Sac_a_dos_app2.borneInf=n.evaluation 
                #mise à jour solution
                self.gener_decision(n)
                Sac_a_dos_app2.decision=pd.Series(Sac_a_dos_app2.decision,index=index_donnees)
                
        else: #noeud non feuille
            n.typeNoeud=1
           
        return n
    
    def gener_decision(self,node):
        """
        génère le nombre d'exemplaires des objets par ordre de l'utilité 
        jusqu'à l'objet situé dans node
        """
        while node.typeNoeud!=-1:
            Sac_a_dos_app2.decision.insert(0,node.nbrExemplaires) #insertion au début (push)
            node=node.pere
            
    
    def cout_branche(self,node):
        """
        calcule le cout (volume) d'une branche à partir de node jusqu'à la racine
        """
        cout=0
        while node.typeNoeud!=-1:
            volume_obj=Sac_a_dos_app2.donnees.loc[node.numeroObjet,'volumes']
            cout+=node.nbrExemplaires*volume_obj
            node=node.pere
        return cout
            
    def ajouterNoeudActif(self,n):
        self.noeudsActifs=self.noeudsActifs.append({'noeuds':n,'eval':n.evaluation},ignore_index=True)
        self.noeudsActifs=self.noeudsActifs.sort_values(by='eval',ascending=False) #stratégie "noeud avec la plus grande évaluation d'abord"
        self.noeudsActifs.reset_index(drop=True,inplace=True)
        
    def retirerNoeudActif(self):
        n=self.noeudsActifs.loc[0,'noeuds']
        self.noeudsActifs=self.noeudsActifs.drop([0],axis=0)
        self.noeudsActifs.reset_index(drop=True,inplace=True)
        n.typeNoeud=0
        return n
        
        
class Node2:
    """
    typeNoeud typeNoeud {1:actif,0:inactif-elagué,-1:racine}
    int numeroObjet :{O,1,2,...}
    int nbrExemplaires 
    int evaluation 
    Node2 pere
    Node2[] fils
    """
    def __init__(self,typeNoeud=-1,numeroObjet=None,nbrExemplaires=None,pere=None):
        self.typeNoeud=typeNoeud
        self.numeroObjet=numeroObjet
        self.nbrExemplaires=nbrExemplaires
        self.pere=pere
        self.fils=[]
        
    def evaluer(self): 
        "meme fonctions que celle vue en cours"
        gain_obj=Sac_a_dos_app2.donnees.loc[self.numeroObjet,'gains']
        volume_obj=Sac_a_dos_app2.donnees.loc[self.numeroObjet,'volumes']
        
        gain_noeud_act=gain_obj*self.nbrExemplaires
        volume_noeud_act=volume_obj*self.nbrExemplaires
        
        pere=self.pere
        gain_noeuds_peres=0
        volume_noeuds_peres=0
        
        while pere.typeNoeud!=-1:
            gain_obj_pere=Sac_a_dos_app2.donnees.loc[pere.numeroObjet,'gains']
            gain_noeuds_peres+=gain_obj_pere*pere.nbrExemplaires
            
            volume_obj_pere=Sac_a_dos_app2.donnees.loc[pere.numeroObjet,'volumes']
            volume_noeuds_peres+=volume_obj_pere*pere.nbrExemplaires
            
            pere=pere.pere
            
        gain_tot=gain_noeud_act+gain_noeuds_peres
        volume_tot=volume_noeud_act+volume_noeuds_peres
        
        index_donnees=Sac_a_dos_app2.donnees.index.values
        index_obj_act=np.nonzero(index_donnees==self.numeroObjet)[0][0]
        if index_obj_act+1==len(index_donnees): #noeud feuille
            self.evaluation=gain_tot
           
        else:
            num_prochain_obj=index_donnees[index_obj_act+1]
            gain_prochain_obj=Sac_a_dos_app2.donnees.loc[num_prochain_obj,'gains']
            volume_prochain_obj=Sac_a_dos_app2.donnees.loc[num_prochain_obj,'volumes']
            self.evaluation=gain_tot+(Sac_a_dos_app2.poids-volume_tot)*gain_prochain_obj/volume_prochain_obj
            

#############################################################  end class B&B sans elagage 



#############################################################  Prog dynamique

def SacADos3(W,filepath) :
    #lecture de fichier des donnees 
    data=pd.read_csv(filepath)
    
    #initialisé le vecteur de gain à 0
    K = [0] * (W+1)
    index = len(data["volumes"])
    
    # nbr est une matrice pour garder trace des objets pris avec le nombre d'exemplaire
    nbr = np.zeros((W+1,index))
    
    for w in range(W+1):
        gain = 0 
        for i in range(0,index-1):
            wi = data["volumes"][i] 
            vi= data["gains"][i]
            if  wi <= w :
                if gain < K[w-wi]+ vi:
                    gain = K[w-wi]+ vi
                    
                    #mettre a jour les objets pris avec le nbr d'exemplaire pour chaque objet dans la matrice nbr
                    for l in range(0,index-1):
                        nbr[w][l]=nbr[w-wi][l]
                    nbr[w][i]+= 1
                    
        K[w]=gain
    # affichage du resultat qui se trouve au dernier ligne    
    for p in range(0,index):
        print("objet",p," : le nombre d'exemplaire est ",int(nbr[W][p]) )
    poid= np.dot(nbr[W],data["volumes"])
    
    return K[W], poid # K[W]= est le gain





############################################################# end Prog dynamique

############################################################## TRI par VAL
class ObjetVal:
    """Les parametres des objets """

    def __init__(self, p, val, ind):
        self.p = p  # Le poids de l'objet
        self.val = val  # La valeur de l'objet
        self.ind = ind  # L'index de l'objet


class TriParValeur:

    @staticmethod
    def ValeurMax(objet, capacite):
        objets = pd.read_csv(objet)  # Les objets
        val = objets.iloc[:, 1].values  # Les valeurs
        p = objets.iloc[:, 0].values  # Les poids
        """Fonction pour trouver la valeur maximale"""
        iVal = []  # La liste des objets ordonnés
        iRes = []  # La liste des objets collectés
        for i in range(len(p)):
            iVal.append(ObjetVal(p[i], val[i], i))

        # Trier les objets selon leurs valeurs dans un ordre décroissant.
        iVal.sort(key=lambda x: x.val, reverse=True)

        # On initialise la ValeurTotal à 0
        Valeurtotal = 0
        # On parcourt tous les objets
        for i in iVal:
            pcourrant = int(i.p)
            Valcourrante = int(i.val)
            if capacite >= pcourrant:
                # on collecte l'objet
                iRes.append(ObjetVal(pcourrant, Valcourrante, i))
            # Tant le poids de l'objet courant est inférieur ou égal à la capacité du sac à dos
            while capacite >= pcourrant:
                # on soustrait son poids de la capacité
                capacite -= pcourrant
                #  on ajoute sa valeur a la Valeurtotal
                Valeurtotal += Valcourrante
        return Valeurtotal, iRes, capacite




################################################################## eend TRI par VAL

################################################################TRI par Densite 


class ObjetVal2:
    """Les paramètres des objets """

    def __init__(self, p, val, ind):
        self.p = p  # Le poids de l'objet
        self.val = val  # La valeur de l'objet
        self.ind = ind  # L'index de l'objet
        self.cout = val / p  # La densité de l'objet

    def __lt__(self, autres):
        return self.cout < autres.cout


class TriParDensite:

    @staticmethod
    def ValeurMax(objet, capacite):
        objets = pd.read_csv(objet)  # Les objets
        val = objets.iloc[:, 1].values  # Les valeurs
        p = objets.iloc[:, 0].values  # Les poids
        """Fonction pour trouver la valeur maximale"""
        iVal = []  # La liste des objets ordonnés
        iRes = []  # La liste des objets collectés
        for i in range(len(p)):
            iVal.append(ObjetVal2(p[i], val[i], i))

        # Trier les objets selon leurs densités dans un ordre décroissant.
        iVal.sort(key=lambda x: x.cout, reverse=True)

        # On initialise la ValeurTotal à 0
        Valeurtotal = 0
        for i in iVal:
            pcourrant = int(i.p)
            Valcourrante = int(i.val)
            if capacite >= pcourrant:
                # on collecte l'objet
                iRes.append(ObjetVal2(pcourrant, Valcourrante, i))
            # tant que le poids de l'objet courant est inférieur ou égal à la capacité du sac à dos
            while capacite >= pcourrant:
                # on soustrait son poids de la capacité
                capacite -= pcourrant
                #  on ajoute sa valeur a la Valeurtotal
                Valeurtotal += Valcourrante
        return Valeurtotal, iRes, capacite




################################################################### end densite 

########################################################################TRI paR POIDS

class TriParPoids:

    @staticmethod
    def ValeurMax(objet, capacite):
        objets = pd.read_csv(objet)  # Les objets
        val = objets.iloc[:, 1].values  # Les valeurs
        p = objets.iloc[:, 0].values  # Les poids
        """Fonction pour trouver la valeur maximale"""
        iVal = []  # La liste des objets ordonnés
        iRes = []  # La liste des objets collectés
        for i in range(len(p)):
            iVal.append(ObjetVal(p[i], val[i], i))

        # Trier les objets selon leurs valeurs dans un ordre croissant.
        iVal.sort(key=lambda x: x.p, reverse=False)

        # On initialise la ValeurTotal à 0
        Valeurtotal = 0
        for i in iVal:
            pcourrant = int(i.p)
            Valcourrante = int(i.val)
            if capacite >= pcourrant:
                # on collecte l'objet
                iRes.append(ObjetVal(pcourrant, Valcourrante, i))
            # Tant que le poids de l'objet courant est inférieur ou égal à la capacité du sac à dos
            while capacite >= pcourrant:
                # on soustrait son poids de la capacité
                capacite -= pcourrant
                #  on ajoute sa valeur a la Valeurtotal
                Valeurtotal += Valcourrante
        return Valeurtotal, iRes, capacite




############################################################################# eend pr POIDS


################################################################### ICA enhanced
class Country:
    "String type_country" #imperialist or colony
    "Dictionnary solution"
    def __init__(self,type_country,solution):
        self.type_country=type_country
        self.solution=solution

class Empire:
    "Country imperialist"
    "Country[] colonies"
    def __init__(self,imperialist):
        self.imperialist=imperialist
        self.colonies=[]
    
    def add_colony(self,colony):
        self.colonies.append(colony)
import pandas as pd
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
   
class ICA:
    "DataFrame data" 
    "Country[] countries"
    "Empire[] emipres"
    
    "int number_of_empires"
    
    "int capacite"
    "int number_of_countries"
    "percentage_of_imperialists"
    "float percentage_of_assimilation"
    "float percentage_of_assimilated_colonies"
    "float percentage_of_revolution"
    "float percentage_of_evolved_colonies"
    "int battle_cycles"
    "int stagnation_factor"
    
    "int[] maximum_exemplaires_number"
    "int previous_best_benefit"
    def __init__(self,capacite,csv_file_path,number_of_countries,percentage_of_imperialists,percentage_of_assimilation,percentage_of_assimilated_colonies,percentage_of_revolution,percentage_of_evolved_colonies,battle_cycles,stagnation_factor=None):
        self.data=pd.read_csv(csv_file_path)
        self.capacite=capacite
        self.number_of_countries=number_of_countries
        self.number_of_empires=int(percentage_of_imperialists*number_of_countries)
        self.countries=[]
        self.empires=[]
        self.start_time=0
        
        self.percentage_of_assimilation=percentage_of_assimilation
        self.percentage_of_assimilated_colonies=percentage_of_assimilated_colonies
        self.percentage_of_revolution=percentage_of_revolution
        self.percentage_of_evolved_colonies=percentage_of_evolved_colonies
        self.battle_cycles=battle_cycles
        self.stagnation_factor=stagnation_factor
        
        self.previous_best_benefit=0
        self.maximum_exemplaires_number=(self.capacite/self.data["volumes"].values).astype(int)
        self.stagnation_level=0
        
        #trier les objets par ordre croissant de l'utilité
        self.data2=self.data.copy()
        self.data2["utility"]=self.data2["gains"]/self.data2["volumes"]
        self.data2=self.data2.sort_values(by="utility", ascending=True)
        #récupérer l'ordre des objets selon l'utilité dans "index"
        self.index=self.data2.index.values
        
        self.nbiter=0
        
        self.start=0
        self.gain=[]
        self.weight=0
    
    def ica(self):
        self.initialisation()
        self.start=perf_counter()
        for i in range(self.battle_cycles):
            self.assimilation()
            print("fin assimilation ",i)
            self.revolution()
            print("fin revolution ",i)
            self.competition()
            print("fin competition ",i)
            best_benefit=self.f(self.countries[0].solution["decimal"])
            print("gain du cycle: ",best_benefit,"\n")
            self.gain.append(best_benefit)
            self.weight=np.dot(self.countries[0].solution["decimal"],self.data["volumes"])
            
            self.nbiter+=1
            if best_benefit==self.previous_best_benefit: #si le gain optimal stagne
                self.stagnation_level+=1 #augmenter le niveau de stagnation
            else: self.stagnation_level=0 #réinitialiser le niveau de stagnation si le gain optimal évolue
            self.previous_best_benefit=best_benefit #garder trace du gain actuel pour le comparer avec le suivant
            if self.percentage_of_revolution+0.05*self.stagnation_level>0.5: #pourcentage de révolution dépasse le seuil (0.7)
                break #sortie de la boucle for avant la fin du nombre de cycle de batailles prévu
        
        
    def initialisation(self):
        #générer les colonnies initiales:
        self.gener_countries()
        #sélectionner les imperialists:
        self.countries.sort(reverse=True,key=lambda country: self.f(country.solution["decimal"]))
        nbre_imperialists=0
        for c in self.countries:
            c.type_country="imperialist"
            nbre_imperialists+=1
            if nbre_imperialists==self.number_of_empires: break
        #générer les empires:
        self.gener_empires()
        self.start_time=perf_counter()
        
    def assimilation(self):
        bin_sol_length=len(self.countries[0].solution["binary"]) #récupérer la longeur d'un chromosome
        for empire in self.empires:
            #explorer les colonies de l'empire pour modifier les gènes de certaines d'entre elles
            for colony in empire.colonies:
                if (np.random.uniform(0,1)<self.percentage_of_assimilated_colonies):
                    #masque de chromosome (si 1: remplacer bit de la colonnie par celui de l'imperialist)
                    mask=(np.random.uniform(0,1,bin_sol_length)<self.percentage_of_assimilation)*1 #if: 1->copy bit from imperialist to colony
                    
                    sol_bin=list(colony.solution["binary"])
                    for i in range(bin_sol_length):
                        if mask[i]==1: #recopier le bit courant de l'imperialist vers la colonnie
                            sol_bin[i]=empire.imperialist.solution["binary"][i] #modifier le bit de la colonnie pour correspondre à celui de l'imperialist
                    sol_bin="".join(sol_bin)
                    
                    sol_dec=self.bin_sol_to_dec_sol(sol_bin) #interpréter le chromosome en solution compréhensible (décimale)
                    if self.verify_constraint(sol_dec)==False: 
                        #altérer la solution obtenue si elle ne satisfait pas la contrainte de capacité du sac à dos
                        sol_dec=self.correct_sol(sol_dec)
                        sol_bin=self.genetic_coding(sol_dec)
                    #sauvegarder la nouvelle solution colonnie
                    colony.solution["binary"]=sol_bin
                    colony.solution["decimal"]=sol_dec        
        
    def revolution(self):
        bin_sol_length=len(self.countries[0].solution["binary"]) #taille du chromosome
        dec_sol_length=len(self.countries[0].solution["decimal"]) #taille de la solution décimale
        for empire in self.empires: #parcourir les empires
            for colony in empire.colonies: #parcourir les colonies de l'empire
                if (np.random.uniform(0,1)<self.percentage_of_evolved_colonies):
                    if self.nbiter==0: #appliquer la diversification génétique à la 1ére itération
                        mask=(np.random.uniform(0,1,bin_sol_length)<0.1)*1 #masque généré avec une proba - if 1: flip bit 
                        sol_bin=list(colony.solution["binary"]) #solution codée en binaire
                        for i in range(bin_sol_length): 
                            if mask[i]==1: #inverser le bit concerné (0->1 ou 1->0)
                                if sol_bin[i]=='0': sol_bin[i]='1' 
                                else: sol_bin[i]='0'
                        sol_bin="".join(sol_bin)
                        sol_dec=self.bin_sol_to_dec_sol(sol_bin) #interpréter le chromosome en solution compréhensible (décimale)
                        if self.verify_constraint(sol_dec)==False: 
                            #altérer la solution obtenue si elle ne satisfait pas la contrainte de capacité du sac à dos
                            sol_dec=self.correct_sol(sol_dec)
                            sol_bin=self.genetic_coding(sol_dec)
                        #sauvegarder la nouvelle solution colonnie
                        colony.solution["binary"]=sol_bin
                        colony.solution["decimal"]=sol_dec
                        
                    else: #nbiter>0: appliquer l'exploration du voisinage pour chaque colonie d'un empire (intensification)
                        #masque généré avec une proba. la proba que masque[i]=1 augmente avec la stagnation du gain
                        mask=(np.random.uniform(0,1,dec_sol_length)<self.percentage_of_revolution+0.02*self.stagnation_level)*1
                        dec_sol=colony.solution["decimal"] #solution codée en décimal
                        for i in range(dec_sol_length):
                            if mask[i]==1: #ajouter ou retrancher 1
                                if np.random.uniform(0,1)<=0.5: #ajouter 1 exemplaire
                                    dec_sol[i]+=1
                                elif dec_sol[i]>0: #retrancher 1 exemplaire
                                    dec_sol[i]-=1
                        if self.verify_constraint(dec_sol)==False: 
                            #altérer la solution obtenue si elle ne satisfait pas la contrainte de capacité du sac à dos
                            sol_dec=self.correct_sol(dec_sol)
                        #sauvegarder la nouvelle solution colonnie
                        colony.solution["decimal"]=dec_sol
                        colony.solution["binary"]=self.genetic_coding(dec_sol)
                        
        #rechercher la meilleure colonnie par empire
        for empire in self.empires:
            best_cost=float("-inf")
            best_colony=None
            for colony in empire.colonies:
                colony_cost=self.f(colony.solution["decimal"])
                if colony_cost>best_cost:
                    best_colony=colony
                    best_cost=colony_cost
            if best_colony!=None:
                #vérifier si la colonnie n'est pas devenu meilleur que l'imperialist
                if self.f(best_colony.solution["decimal"])>self.f(empire.imperialist.solution["decimal"]): 
                    self.colony_ascent(empire,best_colony)
        """        
        for e in self.empires:
            print(e.imperialist.type_country,": ",self.f(e.imperialist.solution["decimal"]),"\n")
            for c in e.colonies:
                print(c.type_country,": ",self.f(c.solution["decimal"]),"\n")
        print("***************************************************************************")
        """                   
    def competition(self): 
        #explorer toutes les colonnies
        for country in self.countries:
            if country.type_country=="colony":
                #empire courant de la colonnie
                current_empire=self.get_empire_of_colony(country) 
                #distance euclidienne entre la solution colonnie et son imperialist
                current_dist=np.linalg.norm(np.array(country.solution["decimal"])-np.array(current_empire.imperialist.solution["decimal"]))
                #vérifier la possiblité que la solution colonnie soit plus proche d'un autre imperialist
                min_dist=float("inf")
                for empire in self.empires:
                    if empire!=current_empire:
                        dist=np.linalg.norm(np.array(country.solution["decimal"])-np.array(empire.imperialist.solution["decimal"]))
                        if dist<min_dist: 
                            nearest_empire=empire
                            min_dist=dist
                #si la colonnie est plus proche d'un autre imperialist que l'imperialist courant
                if min_dist<current_dist and current_empire!=nearest_empire:
                    #la colonnie change d'empire
                    current_empire.colonies.remove(country)
                    nearest_empire.colonies.append(country)
                    
                    #vérifier possibilité que la colonnie soit plus puissante que l'imperialist du nouvel empire
                    if self.f(country.solution["decimal"])>self.f(nearest_empire.imperialist.solution["decimal"]):
                        self.colony_ascent(nearest_empire,country)
        self.remove_weakest_empires() 
                                  
    def colony_ascent(self,empire,colony): #échange les places de l'imperialist de l'empire et la colonie
        old_imperialist=empire.imperialist #sauvegarder l'ancien imperialist
        old_imperialist.type_country="colony"
        colony.type_country="imperialist"
        #print("colony: ",colony,"\n")
        #print("empire.colonies: ",empire.colonies,"\n")
        empire.imperialist=colony #le remplacer par l'ancienne colonnie
        empire.colonies.remove(colony) #supprimer l'ancienne colonnie de la liste des colonnies
        empire.colonies.append(old_imperialist) #ajouter l'ancien imperialist à la liste des colonnies 
        
    def remove_weakest_empires(self): #supprime un imperialist sans colonnie et dont la force n'est pas la meilleure
        #print("****************debut remove***************************")
        self.countries.sort(reverse=True,key=lambda country: self.f(country.solution["decimal"])) #trier les pays par ordre de puissance
        strongest_country_cost=self.f(self.countries[0].solution["decimal"])
        for empire in self.empires:
            #print(empire.colonies)
            if len(empire.colonies)==0 and self.f(empire.imperialist.solution["decimal"])<strongest_country_cost:
                self.empires.remove(empire)
                self.countries.remove(empire.imperialist)
        #print("****************fin remove***************************")
        
    def get_empire_of_colony(self,colony):
        for empire in self.empires:
            if colony in empire.colonies: return empire
            
        
    def gener_random_sols(self,n): #génère n solutions aléatoires
        sols=[]
        for _ in range(n):
            sol=np.random.randint(0,2,self.data.shape[0])
            while self.verify_constraint(sol)==False: 
                #print("bcl")
                sol=self.correct_sol(np.random.randint(0,2,self.data.shape[0]))
            sols.append(sol)
        return sols
            
    def gener_countries(self): #génère les pays initiaux comme étant des colonies
        initial_solutions={}
        initial_solutions["decimal"]=[]
        initial_solutions["binary"]=[]
        #générer un ensemble de solutions initiales avec recuit simulé
        initial_solutions["decimal"]=self.gener_random_sols(self.number_of_countries)
        #coder les solutions initiales en binaire (codage génétique)
        initial_solutions["binary"]=[self.genetic_coding(decimal_sol) for decimal_sol in initial_solutions["decimal"]]
        for i in range(self.number_of_countries):
            solution={"decimal":initial_solutions["decimal"][i],"binary":initial_solutions["binary"][i]}
            self.countries.append(Country("colony",solution))
    
    def gener_empires(self):
        for i in range(self.number_of_empires):
            self.empires.append(Empire(self.countries[i]))
        dist_min=float("inf")
        for c in self.countries:
            if c.type_country=="colony":
                for empire in self.empires:
                    dist=np.linalg.norm(np.array(c.solution["decimal"])-np.array(empire.imperialist.solution["decimal"]))
                    if dist<dist_min:
                        nearest_empire=empire
                        dist_min=dist
                dist_min=float("inf")
                nearest_empire.add_colony(c)
        """
        for e in self.empires:
            print(e.imperialist.type_country,": ",self.f(e.imperialist.solution["decimal"]),"\n")
            for c in e.colonies:
                print(c.type_country,": ",self.f(c.solution["decimal"]),"\n")
        print("************************************************************************************")
        """
        
    def genetic_coding(self,decimal_array_solution):
        binary_solution=""
        for (nbr_exemplaires_chosen,nbr_exemplaires_max) in zip(decimal_array_solution,self.maximum_exemplaires_number):
            if nbr_exemplaires_max==0: binary_solution+="0"
            else:
                n=bin(nbr_exemplaires_chosen)[2:]
                while len(n)<int(np.log2(nbr_exemplaires_max))+1: n="0"+n
                binary_solution+=n
        return binary_solution
    
    def bin_sol_to_dec_sol(self,bin_sol):
        dec_sol=[]
        j=0
        binary_number=""
        for n in self.maximum_exemplaires_number:
            if n==0: nbre_bits=1
            else: nbre_bits=int(np.log2(n))+1
            for _ in range(nbre_bits):
                binary_number+=bin_sol[j]
                j+=1
            dec_sol.append(int(binary_number,2))
            binary_number=""
        return dec_sol
    
    def correct_sol(self,sol_dec):
        num_obj=self.index[0]
        i=0
        #décrémenter le nombre d'exemplaires d'objets les moins utiles jusqu'à satisfaire la contrainte de capacité
        while self.verify_constraint(sol_dec)==False:
            while sol_dec[num_obj]==0: 
                i+=1
                num_obj=self.index[i]
            sol_dec[num_obj]-=1   
        return sol_dec
                           
    def f(self,s):
        return np.dot(s,self.data["gains"])
    
    def verify_constraint(self,s):
        return np.dot(s,self.data["volumes"])<=self.capacite
   
     



########################################################################ennd ICA enhanced



################################################################### Recuit SIMULE

class metaheuristique_Voisiange:
    
    def __init__(self,capacite,csv_file_path,aleatoire, alpha,temperature, poids_final = None,
                 s_courant = None,gains = None, f_s_courant = None, voisins=None, itera = None):
    
        self.data=pd.read_csv(csv_file_path, sep=",") #la liste des objets définis par le couple (volume,gain)
        self.capacite = capacite #la capacité maximale du sac à dos
        self.s_courant = s_courant  #la solution courante
        self.f_s_courant = f_s_courant #la valeur de la fonction objectif pour la solution courante
        self.temperature =[] #la liste des températures
        self.temperature.append(temperature) 
        self.voisins = [] #la liste des voisins visités
        self.aleatoire = aleatoire #choisir une solution initiale triviale (0,0,..) ou une solution initiale aléatoire
        self.alpha = alpha #le facteur de refroidissement
        self.gains = [] #la liste des gains
        self.itera = itera #le nombre d'itérations nécessaires pour trouver la solution
        self.poids_final = poids_final #le poids final des objets ajoutés au sac à dos
        
    def recuit_simule(self):
        #Initialisation
        if (self.aleatoire == True):       #la solution initiale
            self.s_courant = self.generer_sol_aleat() #aléatoire
        else:
            self.s_courant = np.zeros(self.data.shape[0]) #triviale
       
        self.voisins.append(self.s_courant)
        self.gains.append(self.f(self.s_courant))
        
        nbr_iter = 500 #le nombre d'itérations
        
        t = self.temperature[0]
        
        i=0
        continuer = True
        #continuer = faux: arreter la recherche lorsque la voisine est egale à la solution_courante
        while ( i < nbr_iter) and (continuer == True):
            
            s , continuer = self.voisin (self.s_courant)   #une solution voisine
            if (continuer == True):
                delta = self.f(s) - self.f(self.s_courant)

                if (delta > 0): #accepter la solution qui améliore f sans aucune condition
                    self.s_courant = s
                    self.voisins.append(self.s_courant)
                    self.gains.append(self.f(self.s_courant))
                else:
                    mu = np.random.rand()
                    if (mu < np.exp(delta / t)): #accepter une solution qui n'améliore pas f avec condition
                        self.s_courant = s
                        self.voisins.append(self.s_courant)
                        self.gains.append(self.f(self.s_courant))
                        
                t *= self.alpha   #refroidissement continu
                self.temperature.append(t)
                i +=1
            else: #stagnation de la solution courante   #La solution ne s'améliore pas apres la i eme itération 
                self.itera = i
        self.f_s_courant = self.f(self.s_courant)
        self.poids_final = np.dot(self.s_courant, self.data['volumes'])  
        self.capacite -= self.poids_final

    def f(self,s): #fonction objectif
        return np.dot(s , self.data['gains'])   #somme (x_i * gain_i)


    def generer_sol_aleat(self):
       
    #générer une solution aléatoire parmi les solutions qui remplissent le sac a dos 
        m = self.data.shape[0]   #nombre d'objets
        sol = np.zeros(m)   #tous les objets initialement à 0 (0 exemplaires)
        capa = 0   #volume cumulé des objets dans le sac à dos
        
        while (capa <= self.capacite):#tant que on n'est pas arrivé à la capacité max
           
            i = np.random.randint(m)   #choisir un objet aléatoirement
            sol[i] += 1   #ajouter un exemplaire de l'objet i au sac à dos
            capa = np.dot(sol.T, self.data['volumes']) #MAJ de la capacité 
            
            if (capa > self.capacite): #si on a dépassé le max, on enleve le dernier exemplaire ajouté et on sort
                sol[i] -= 1
                break   #la solution aléatoire est trouvé
        return sol
    
    
    def voisin (self,s):#retourner la 1ere solution viosine trouvée
       
        continuer = True  #on pourra rajouter d'autres objets 
        trouv = False   #on a pas encore trouver la voisine
        v = s   #v:la solution voisine à s      
        m = self.data.shape[0]
        liste = list(range(m))    #les rangs des objets qu'on peut ajouter au sac a dos
        
        ind = np.random.choice(liste) #enlever un exemplaire de l'objet ind s'il y on a
        if (v[ind] !=0): 
            v[ind] -=1
              
        i = np.random.choice(liste)     #rang de l'objet
        
        while ((trouv == False) and (continuer == True)):

            v[i] += 1   #ajouter un exemplaire de l'objet i

            if (np.dot(v.T, self.data['volumes']) < self.capacite): #somme (x_i * volume_i)
                trouv = True   #s'il verifie la contraine de poids donc c'est une voisine
            else: 
                v[i] -= 1
                liste.remove(i) #elimener l'objet i de liste
              
                if (len(liste) == 0): 
                    continuer = False   #on ne pourra pas rajouter d'autres objets 
                else: i = np.random.choice(liste)

        return v , continuer
              
########################################################################ennd Recuit SIMULE



################################################################### Fireworks
class Firework:
    "Int amplitude" #The explosion amplitude of each firework 
    "Int nb_sparks"    #the number of explosion sparks of each firework 
    "Dictionnary Value"
    "Dictionnary[] sparks"
    def __init__(self,amplitude,sparks,Value):
        self.nb_sparks=sparks
        self.Value=Value
        self.amplitude=amplitude
        self.sparks=[]
    def add_spark(self,spark):
        self.sparks.append(spark)


import pandas as pd
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
import random
import time
   
class FireWorks:
    "DataFrame data"
    "int capacite"
    "Firework[] fireworks"

    "int nb_sparks"
    "int amplitude"
    "int mutation"
    "int I"
    "int N"
    
    "int[] maximum_exemplaires_number"
    def __init__(self,capacite,csv_file_path,nb_sparks,amplitude,nb_mutation,N,I):
        self.path=csv_file_path
        self.data=pd.read_csv(csv_file_path)
        self.capacite=capacite
        self.nb_sparks=nb_sparks
        self.amplitude=amplitude
        self.N=N
        self.nb_mutation=nb_mutation
        self.I=I
        self.fireworks=[]
        self.solutions=[]
        self.newsolutions=[]
        self.gain=[]
        #self.maximum_exemplaires_number=(self.capacite/self.data["volumes"].values).astype(int)
    
    
    def calcul_amplitude(self,i,initial_solutions): #calcul l'attribut sparks d'un feux d'artifice
        sumi=np.dot(self.f(initial_solutions),np.ones(len(initial_solutions)))
        s=self.amplitude*self.f(initial_solutions[i])/sumi
        return int(round(s))
        
    
    def calcul_sparks(self,i,initial_solutions): # calcul l'attribut amplitude d'un feux d'artifice lambda i
        sumi=np.dot(self.f(initial_solutions),np.ones(len(initial_solutions)))
        s=self.nb_sparks*self.f(initial_solutions[i])/sumi
        return int(round(s))
        
    
    def generer_sol_aleat(self):
       
    #générer une solution aléatoire parmi les solutions qui remplissent le sac a dos 
        m = self.data.shape[0]; #nombre d'objets
        sol = np.zeros(m); #tous les objets initialement à 0 (0 exemplaires)
        capa = 0; #volume cumulé des objets dans le sac à dos
        
        while (capa <= self.capacite):#tant que on n'est pas arrivé à la capacité max
           
            i = np.random.randint(m); #choisir un objet aléatoirement
            sol[i] += 2; #ajouter un exemplaire de l'objet i au sac à dos
            capa = np.dot(sol.T, self.data['volumes']) #MAJ de la capacité 
            
            if (capa > self.capacite): #si on a dépassé le max, on enleve le dernier exemplaire ajouté et on sort
                sol[i] -= 1
                break; #la solution aléatoire est trouvé
        if self.verify_constraint(sol )==False: 
                   #Mapping to aceptable
                   newsol=self.correct_sol(sol)
        return sol
    
    
    def gener_random_sols(self,n): #génère n solutions aléatoires
        sols=[]
       # app = metaheuristique_Voisiange(13570,"demo instance.csv",False,0.9,900)            
        #app.recuit_simule()   
        #sols.append(app.s_courant)
        #n=n-1
        for _ in range(n):
            #print("raaandom ",_)
            #sol=np.random.randint(0,2,self.data.shape[0])
            sol=self.generer_sol_aleat()
            sols.append(sol)
        return sols
        
   
    
    
    
    def correct_sol(self,sol_dec):
        #trier les objets par ordre croissant de l'utilité
        data2=self.data.copy()
        data2["utility"]=data2["gains"]/data2["volumes"]
        data2=data2.sort_values(by="utility", ascending=True)
        
        #récupérer l'ordre des objets selon l'utilité dans "index"
        index=data2.index.values
        
        num_obj=index[0]
        i=0
        #décrémenter le nombre d'exemplaires d'objets les moins utiles jusqu'à satisfaire la contrainte de capacité
        while self.verify_constraint(sol_dec)==False:
            while sol_dec[num_obj]==0: 
                i+=1
                num_obj=index[i]
            sol_dec[num_obj]-=1
            
        return sol_dec
                           
    def f(self,s):
        return np.dot(s,self.data["gains"])
    
    def verify_constraint(self,s):
        return np.dot(s,self.data["volumes"])<=self.capacite
   
    def getsolutions(self):   
        for _ in self.fireworks:
            self.solutions.append(_.Value)
            for __ in _.sparks:
                self.solutions.append(__.Value)
    def getbestsol(self):
        
        return np.argmax(self.f(self.solutions))
    ###########################################
    
    
    def algofireworks(self):
        self.population()
        
        #print(self.f(self.fireworks[7].Value["decimal"]))
        for _ in range(self.I):
            print("--- exploode ---")
            self.explosion()
            print("--- mutat ---")
            #print(len(self.fireworks[7].sparks))
            self.mutation()
            print("--- select ---")
            self.selection()
            print("--- next IIT ---")
        self.show()
            
    def population(self):
        
        initial_solutions={}
        initial_solutions=[]
        print("--- debut population ---")
        #générer un ensemble de solutions initiales 
        initial_solutions=self.gener_random_sols(self.N)
        #just choose the best
        max_index_row = np.argmax(self.f(initial_solutions))
        #maximum=np.max(self.f(initial_solutions))
        
        self.solutions=[]
        
        self.solutions.append(initial_solutions[max_index_row])
        
        ####heuristique sol
        #app = metaheuristique_Voisiange(self.capacite,self.path,False,0.9,900)            

        #app.recuit_simule()
       
       
        #self.solutions.append(app.s_courant)
        #initial_solutions={}
        #initial_solutions=[]
        
        #générer un ensemble de solutions initiales 
        self.solutions=self.gener_random_sols(self.N)
        n=(self.N)-1
        for i in range(n):
            self.solutions.append(initial_solutions[i])
        #initial_solutions=self.gener_random_sols(self.N)
        #just choose the best
        #max_index_row = np.argmax(self.f(initial_solutions))
        #maximum=np.max(self.f(initial_solutions))
        
        
        #self.solutions.append(initial_solutions[max_index_row])
       
        
        #print(self.solutions)
        
        #for i in range(2):
         #   solution=initial_solutions[i]
          #  self.solutions.append(solution)
#
 #           amplitude=self.calcul_amplitude(i,initial_solutions)
      #      sparks=self.calcul_sparks(i,initial_solutions)
       #     self.fireworks.append(Firework(amplitude,sparks,solution))
                #print(sparks,amplitude)

    
    def explosion(self):
        #print("ensemble lewell ",self.solutions)
        for i in range(self.N):
            amplitude=self.calcul_amplitude(i,self.solutions)
            sparks=self.calcul_sparks(i,self.solutions)
            for j in range(self.nb_sparks):
                e=round(amplitude *np.random.random())
                #print(e)
                d=round(self.data.shape[0])
                z=round(d*np.random.random()) 
                #newfirework= Firework(firework.amplitude,firework.sparks,firework.Value) 
                newsol=self.solutions[i].copy()
                
                #print(firework.Value)
                for _ in range(z)  :
                    b=random.randint(0, d-1)
                    newsol[b]=newsol[b]+e
                if self.verify_constraint(newsol )==False: 
             #Mapping to aceptable
                   newsol=self.correct_sol(newsol)
                
                self.solutions.append(newsol)
               
                #print("ensemble +1",self.solutions)
                
                #firework.add_spark(newfirework)    
                #print(newfirework.Value,self.f(newfirework.Value["decimal"]))
    
    
    def mutation(self):
        
        for i in range(self.nb_mutation):
            
            #np.random.shuffle(self.fireworks)
            #firework=random.choice(self.fireworks)
            #print(len(firework.sparks))
            np.random.shuffle(self.solutions)
            solution=self.solutions[i]
           
            amplitude=self.calcul_amplitude(i,self.solutions)
            sparks=self.calcul_sparks(i,self.solutions)
            e=round(amplitude *np.random.random())
                #print(e)
            d=round(self.data.shape[0])
            z=round(d*np.random.random()) 
            #newfirework= Firework(firework.amplitude,firework.sparks,firework.Value) 
            newsol=self.solutions[i].copy()
            #print(firework.Value)
            for _ in range(z)  :
                b=round(random.randint(0, d-1))
                newsol[b]=newsol[b]+e +2
                if self.verify_constraint(newsol )==False: 
                   #Mapping to aceptable
                   newsol=self.correct_sol(newsol)
                self.solutions.append(newsol)
                #firework.add_spark(newfirework)    
                #print(self.f(firework.sparks[3].Value["decimal"]))
            
        

    def selection(self):
        #self.getsolutions()
        #print(self.solutions)
        maxim=self.getbestsol()
        self.newsolutions=[]
        self.newsolutions.append(self.solutions[maxim])
        print("gain de l'iterztion ",self.f(self.solutions[maxim]))
        self.gain.append(self.f(self.solutions[maxim]))
        #print(self.solutions[maxim])
        #print(self.newsolutions)
       
        
        n=(self.N)-1
        for i in range(n):
            np.random.shuffle(self.solutions)
            solution=random.choice(self.solutions)
            self.newsolutions.append(solution)
        
        #print(self.newsolutions)
        self.solutions=self.newsolutions.copy()    
        #self.fireworks=[]
        #for i in range(2):
            #solution=self.newsolutions[i]
            
            #amplitude=self.calcul_amplitude(i, self.newsolutions)
            #sparks=self.calcul_sparks(i, self.newsolutions)
            #self.fireworks.append(Firework(amplitude,sparks,solution))
        
        
        
        
    def show(self):    
        #show best weight and bonus
        maxim=self.getbestsol()
        print("gain optimal : ",self.f(self.solutions[maxim]))
        return 0
########################################################################ennd fireworks

############################################################## hybride L

   
class LRH:
    "DataFrame data" 
    "Country[] countries"
    "Empire[] emipres"
    
    "int number_of_empires"
    
    "int capacite"
    "int number_of_countries"
    "percentage_of_imperialists"
    "float percentage_of_assimilation"
    "float percentage_of_assimilated_colonies"
    "float percentage_of_RC"
    "float percentage_of_evolved_colonies"
    "int battle_cycles"
    "int stagnation_factor"
    
    "int[] maximum_exemplaires_number"
    "int previous_best_benefit"
    def __init__(self,capacite,csv_file_path,number_of_countries,percentage_of_imperialists,percentage_of_assimilation,percentage_of_assimilated_colonies,percentage_of_RC,percentage_of_evolved_colonies,battle_cycles,temperature, alpha):
        self.data=pd.read_csv(csv_file_path)
        self.capacite=capacite
        self.number_of_countries=number_of_countries
        self.number_of_empires=int(percentage_of_imperialists*number_of_countries)
        self.countries=[]
        self.empires=[]
        
        self.percentage_of_assimilation=percentage_of_assimilation
        self.percentage_of_assimilated_colonies=percentage_of_assimilated_colonies
        self.percentage_of_RC=percentage_of_RC
        self.percentage_of_evolved_colonies=percentage_of_evolved_colonies
        self.battle_cycles=battle_cycles
        self.temperature= temperature
        self.alpha=alpha
        
        self.previous_best_benefit=0
        self.gain=[]
        self.weight=0
        self.maximum_exemplaires_number=(self.capacite/self.data["volumes"].values).astype(int)
        self.stagnation_level=0
        
        #trier les objets par ordre croissant de l'utilité
        self.data2=self.data.copy()
        self.data2["utility"]=self.data2["gains"]/self.data2["volumes"]
        self.data2=self.data2.sort_values(by="utility", ascending=True)
        #récupérer l'ordre des objets selon l'utilité dans "index"
        self.index=self.data2.index.values
        
        self.start=0
    
    def lrh(self):
        self.initialisation()
        self.start=perf_counter()
        for i in range(self.battle_cycles):
            self.assimilation()
            print("fin assimilation ",i)
            self.RC()
            print("fin revolution ",i)
            self.competition()
            print("fin competition ",i)
            best_benefit=self.f(self.countries[0].solution["decimal"])
            print("gain du cycle: ",best_benefit,"\n")
            self.gain.append(best_benefit)
            
            if best_benefit==self.previous_best_benefit: #si le gain optimal stagne
                self.stagnation_level+=1 #augmenter le niveau de stagnation
            else: self.stagnation_level=0 #réinitialiser le niveau de stagnation si le gain optimal évolue
            self.previous_best_benefit=best_benefit #garder trace du gain actuel pour le comparer avec le suivant
            if self.percentage_of_RC+0.025*self.stagnation_level>0.5: #pourcentage de révolution dépasse le seuil (0.7)
                break #sortie de la boucle for avant la fin du nombre de cycle de batailles prévu
        self.weight=np.dot(self.countries[0].solution["decimal"],self.data["volumes"])
            

        
        
    def initialisation(self):
        #générer les colonnies initiales:
        self.gener_countries()
        #sélectionner les imperialists:
        self.countries.sort(reverse=True,key=lambda country: self.f(country.solution["decimal"]))
        nbre_imperialists=0
        for c in self.countries:
            c.type_country="imperialist"
            nbre_imperialists+=1
            if nbre_imperialists==self.number_of_empires: break
        #générer les empires:
        self.gener_empires()
        self.start_time=perf_counter()
        
    def assimilation(self):
        bin_sol_length=len(self.countries[0].solution["binary"]) #récupérer la longeur d'un chromosome
        for empire in self.empires:
            #explorer les colonies de l'empire pour modifier les gènes de certaines d'entre elles
            for colony in empire.colonies:
                if (np.random.uniform(0,1)<self.percentage_of_assimilated_colonies):
                    #masque de chromosome (si 1: remplacer bit de la colonnie par celui de l'imperialist)
                    mask=(np.random.uniform(0,1,bin_sol_length)<self.percentage_of_assimilation)*1 #if: 1->copy bit from imperialist to colony
                    
                    sol_bin=list(colony.solution["binary"])
                    for i in range(bin_sol_length):
                        if mask[i]==1: #recopier le bit courant de l'imperialist vers la colonnie
                            sol_bin[i]=empire.imperialist.solution["binary"][i] #modifier le bit de la colonnie pour correspondre à celui de l'imperialist
                    sol_bin="".join(sol_bin)
                    
                    sol_dec=self.bin_sol_to_dec_sol(sol_bin) #interpréter le chromosome en solution compréhensible (décimale)
                    if self.verify_constraint(sol_dec)==False: 
                        #altérer la solution obtenue si elle ne satisfait pas la contrainte de capacité du sac à dos
                        sol_dec=self.correct_sol(sol_dec)
                        sol_bin=self.genetic_coding(sol_dec)
                    #sauvegarder la nouvelle solution colonnie
                    colony.solution["binary"]=sol_bin
                    colony.solution["decimal"]=sol_dec        
        
    def RC(self):
        bin_sol_length=len(self.countries[0].solution["binary"])
        dec_sol_length=len(self.countries[0].solution["decimal"])
        
        nbr_itr = dec_sol_length
        t=self.temperature 
        alpha=self.alpha
        
        for empire in self.empires:
            for colony in empire.colonies:
                
                i=0
                if (np.random.uniform(0,1)<self.percentage_of_evolved_colonies):
                    mask=(np.random.uniform(0,1,dec_sol_length)<self.percentage_of_RC+0.02*self.stagnation_level)*1
                    dec_sol=colony.solution["decimal"]
                    s=dec_sol
                    while i< nbr_itr :
                        if mask[i]==1: #ajouter ou retrancher 1
                            if np.random.uniform(0,1)<=0.5: dec_sol[i]+=1
                            elif dec_sol[i]>0: dec_sol[i]-=1
                        i += 1
                        
                        
                    if self.verify_constraint(dec_sol)==False: 
                        dec_sol=self.correct_sol(dec_sol)
                    
                        
                    delta = self.f(dec_sol) - self.f(s)
                    if (delta > 0): #accepter la solution qui améliore f sans aucune condition
                        s = dec_sol;   
                    else:
                        mu = np.random.rand()
                        if (mu < np.exp(delta/t)):
                            s = dec_sol
                        
                        
                    t *= alpha; #refroidissement continu   
                    
                    #sauvegarder la nouvelle solution colonnie
                    colony.solution["decimal"]=s
                    colony.solution["binary"]=self.genetic_coding(s)
                    
        #rechercher la meilleure colonnie par empire
        for empire in self.empires:
            best_cost=float("-inf")
            best_colony=None
            for colony in empire.colonies:
                colony_cost=self.f(colony.solution["decimal"])
                if colony_cost>best_cost:
                    best_colony=colony
                    best_cost=colony_cost
            if best_colony!=None:
                #vérifier si la colonnie n'est pas devenu meilleur que l'imperialist
                if self.f(best_colony.solution["decimal"])>self.f(empire.imperialist.solution["decimal"]): 
                    self.colony_ascent(empire,best_colony)
      
    
    def competition(self): 
        #explorer toutes les colonnies
        for country in self.countries:
            if country.type_country=="colony":
                #empire courant de la colonnie
                current_empire=self.get_empire_of_colony(country) 
                #distance euclidienne entre la solution colonnie et son imperialist
                current_dist=np.linalg.norm(np.array(country.solution["decimal"])-np.array(current_empire.imperialist.solution["decimal"]))
                #vérifier la possiblité que la solution colonnie soit plus proche d'un autre imperialist
                min_dist=float("inf")
                for empire in self.empires:
                    if empire!=current_empire:
                        dist=np.linalg.norm(np.array(country.solution["decimal"])-np.array(empire.imperialist.solution["decimal"]))
                        if dist<min_dist: 
                            nearest_empire=empire
                            min_dist=dist
                #si la colonnie est plus proche d'un autre imperialist que l'imperialist courant
                if min_dist<current_dist and current_empire!=nearest_empire:
                    #la colonnie change d'empire
                    current_empire.colonies.remove(country)
                    nearest_empire.colonies.append(country)
                    
                    #vérifier possibilité que la colonnie soit plus puissante que l'imperialist du nouvel empire
                    if self.f(country.solution["decimal"])>self.f(nearest_empire.imperialist.solution["decimal"]):
                        self.colony_ascent(nearest_empire,country)
        self.remove_weakest_empires() 
                                  
    def colony_ascent(self,empire,colony): #échange les places de l'imperialist de l'empire et la colonie
        old_imperialist=empire.imperialist #sauvegarder l'ancien imperialist
        old_imperialist.type_country="colony"
        colony.type_country="imperialist"
        #print("colony: ",colony,"\n")
        #print("empire.colonies: ",empire.colonies,"\n")
        empire.imperialist=colony #le remplacer par l'ancienne colonnie
        empire.colonies.remove(colony) #supprimer l'ancienne colonnie de la liste des colonnies
        empire.colonies.append(old_imperialist) #ajouter l'ancien imperialist à la liste des colonnies 
        
    def remove_weakest_empires(self): #supprime un imperialist sans colonnie et dont la force n'est pas la meilleure
        #print("****************debut remove***************************")
        self.countries.sort(reverse=True,key=lambda country: self.f(country.solution["decimal"])) #trier les pays par ordre de puissance
        strongest_country_cost=self.f(self.countries[0].solution["decimal"])
        for empire in self.empires:
            #print(empire.colonies)
            if len(empire.colonies)==0 and self.f(empire.imperialist.solution["decimal"])<strongest_country_cost:
                self.empires.remove(empire)
                self.countries.remove(empire.imperialist)
        #print("****************fin remove***************************")
        
    def get_empire_of_colony(self,colony):
        for empire in self.empires:
            if colony in empire.colonies: return empire
            
        
    def gener_random_sols(self,n): #génère n solutions aléatoires
        sols=[]
        for _ in range(n):
            sol=np.random.randint(0,2,self.data.shape[0])
            while self.verify_constraint(sol)==False: 
                #print("bcl")
                sol=self.correct_sol(np.random.randint(0,2,self.data.shape[0]))
            sols.append(sol)
        return sols
            
    def gener_countries(self): #génère les pays initiaux comme étant des colonies
        initial_solutions={}
        initial_solutions["decimal"]=[]
        initial_solutions["binary"]=[]
        #générer un ensemble de solutions initiales avec recuit simulé
        initial_solutions["decimal"]=self.gener_random_sols(self.number_of_countries)
        #coder les solutions initiales en binaire (codage génétique)
        initial_solutions["binary"]=[self.genetic_coding(decimal_sol) for decimal_sol in initial_solutions["decimal"]]
        for i in range(self.number_of_countries):
            solution={"decimal":initial_solutions["decimal"][i],"binary":initial_solutions["binary"][i]}
            self.countries.append(Country("colony",solution))
    
    def gener_empires(self):
        for i in range(self.number_of_empires):
            self.empires.append(Empire(self.countries[i]))
        dist_min=float("inf")
        for c in self.countries:
            if c.type_country=="colony":
                for empire in self.empires:
                    dist=np.linalg.norm(np.array(c.solution["decimal"])-np.array(empire.imperialist.solution["decimal"]))
                    if dist<dist_min:
                        nearest_empire=empire
                        dist_min=dist
                dist_min=float("inf")
                nearest_empire.add_colony(c)
     
        
    def genetic_coding(self,decimal_array_solution):
        binary_solution=""
        for (nbr_exemplaires_chosen,nbr_exemplaires_max) in zip(decimal_array_solution,self.maximum_exemplaires_number):
            if nbr_exemplaires_max==0: binary_solution+="0"
            else:
                n=bin(nbr_exemplaires_chosen)[2:]
                while len(n)<int(np.log2(nbr_exemplaires_max))+1: n="0"+n
                binary_solution+=n
        return binary_solution
    
    def bin_sol_to_dec_sol(self,bin_sol):
        dec_sol=[]
        j=0
        binary_number=""
        for n in self.maximum_exemplaires_number:
            if n==0: nbre_bits=1
            else: nbre_bits=int(np.log2(n))+1
            for _ in range(nbre_bits):
                binary_number+=bin_sol[j]
                j+=1
            dec_sol.append(int(binary_number,2))
            binary_number=""
        return dec_sol
    
    def correct_sol(self,sol_dec):
        num_obj=self.index[0]
        i=0
        #décrémenter le nombre d'exemplaires d'objets les moins utiles jusqu'à satisfaire la contrainte de capacité
        while self.verify_constraint(sol_dec)==False:
            #print("4.1 \n")
            while sol_dec[num_obj]==0: 
                i+=1
                num_obj=self.index[i]
            sol_dec[num_obj]-=1   
            #print("4.2 \n")
        return sol_dec
                           
    def f(self,s):
        return np.dot(s,self.data["gains"])
    
    def verify_constraint(self,s):
        return np.dot(s,self.data["volumes"])<=self.capacite

################################################################## eend htbride L


############################################################## hybride H



class metaheuristique_Voisiange2:
    
    def __init__(self, capacite, csv_file_path, s_initial ,alpha,temperature ):
        self.data=pd.read_csv(csv_file_path); 
        self.capacite = capacite; 
        self.s_courant = s_initial;
        self.t= temperature
        self.temperature = []
        self.voisins = []
        self.alpha = alpha
        self.gains = []
        self.poid = 0
    def recuit_simule(self):

        
        self.voisins.append(self.s_courant);
        self.gains.append(self.f(self.s_courant))

        nbr_iter = 500;
        self.temperature.append(self.t);
        
        
        i=0;
        continuer = True;
        #continuer = faux: arreter la recherche lorsque la voisine est egale à la solution_courante donc elle est optimale
        while ( i < nbr_iter) and (continuer == True):

            s , continuer = self.voisin (self.s_courant); #une solution voisine
            if (continuer == True):
                delta = self.f(s) - self.f(self.s_courant);

                if (delta > 0): #accepter la solution qui améliore f sans aucune condition
                    self.s_courant = s;
                    self.voisins.append(self.s_courant);
                    self.gains.append(self.f(self.s_courant))
                else:
                    mu = np.random.rand();
                    if (mu < np.exp(delta/self.t)):
                        self.s_courant = s;
                        self.voisins.append(self.s_courant);
                        self.gains.append(self.f(self.s_courant))
                        
                self.t *= self.alpha; #refroidissement continu
                self.temperature.append(self.t);
                i +=1;
            else:
                print("La solution ne s'améliore pas apres la ", i ," eme itération")

        self.f_s_courant = self.f(self.s_courant);
        self.capacite -= np.dot(self.s_courant, self.data['volumes']); 
        self.poid = np.dot(self.s_courant, self.data['volumes'])

    def f(self,s): #fonction objectif
        return np.dot(s , self.data['gains']); #somme (x_i * gain_i)


    def generer_sol_aleat(self):
        #générer une solution qui remplit le sac a dos 
        m = self.data.shape[0];
        sol = np.zeros(m);
        capa = 0;
        
        while (capa <= self.capacite):
            i = np.random.randint(m);
            sol[i] += 1;
            capa = np.dot(sol.T, self.data['volumes']) 
            if (capa > self.capacite): 
                sol[i] -= 1;
                break;
        return sol;
    
    
    def voisin (self,s):        #retourner la 1ere solution viosine trouvée en rajoutant un exemplaire d'un objet
       
        continuer = True;#on pourra rajouter d'autres objets 
        trouv = False; #on a pas encore trouver la voisine
        v = s; #v:la solution voisine à s      
        m = self.data.shape[0];
        liste = list(range(m));  #les rangs des objets qu'on peut ajouter au sac a dos
        ind = np.random.choice(liste)
        if (v[ind] !=0): 
            v[ind] -=1;
              
        i = np.random.choice(liste);   #rang de l'objet
        while ((trouv == False) and (continuer == True)):
            v[i] += 1; #ajouter un exemplaire de l'objet i
            k = np.array(v)
            if (np.dot(k.T, self.data['volumes']) < self.capacite): #somme (x_i * volume_i)
                trouv = True; #s'il verifie la contraine de poids donc c'est une voisine
            else: 
                v[i] -= 1;    
                liste.remove(i); #elimener l'objet i de liste
                if (len(liste) == 0): 
                    continuer = False; #on ne pourra pas rajouter d'autres objets 
                else: i = np.random.choice(liste)
        return v , continuer;




################################################################## eend htbride H


######################## ACTUAL VIEWS 

def upload(request):
    ##getting file
    for key, uploaded_file in request.FILES.items():
        if request.method == 'POST' :
            methode=request.POST['methode']
            poids=int(request.POST['poids'])
            
            uploaded = request.FILES['document']
            fs = FileSystemStorage()
            filename = fs.save(uploaded.name, uploaded_file)
            #uploaded_file_url = fs.url(filename)
            #print(uploaded_file_url)
            ########launching algo 
            path = uploaded_file.name
            dest = open(path, 'wb')
            if uploaded_file.multiple_chunks:
                for c in uploaded_file.chunks():
                    dest.write(c)
                else:
                    dest.write(uploaded_file.read())
                dest.close()    
            tempsexec=0
            gain=0
            
            if(methode=="Prog Dynamique"):
                execute= progdyn
           
            elif (methode=="B&B -avec Elagage"):
                execute= bbavecelagage

            elif (methode=="B&B -sans Elagage"):
                execute= bbsanselagage
            
            elif (methode=="Fireworks"):
                execute= fireworks

            elif (methode=="Imperialist cometitive"):
                execute= ica

            elif (methode=="Methode Hybride -L"):
                execute= Hybridel    
            
            elif (methode=="Methode Hybride -H"):
                execute= Hybrideh 

            elif (methode=="Tri par Val"):
                execute= triparval 

            elif (methode=="Tri par Poids"):
                execute= triparpoids

            elif (methode=="Tri par Densite"):
                execute= tripardensite   
            
            elif (methode=="Recuit Simule"):
                execute= recuitsimule  
            else:
                print("PROLEME AVEC LE NOM DE LA  METHODE")
            
            tempsexec,gain,poids  =execute(request,poids,filename)



            return render(request, 'algoritmmanagement/upload.html', {
                'meth' : methode,
                'uploaded_file_url': fs.url(filename),
                'tempsexec': tempsexec,
                'gain': gain,
                'poids': poids

            })
    return render(request, 'algoritmmanagement/upload.html')


            

# Create your views here.
def bbavecelagage(request,poids,filename):
    fs = FileSystemStorage()
    app=Sac_a_dos_app(poids,'.'+fs.url(filename)) 
    start_time=perf_counter()           
    app.lancer()
    temps=  perf_counter() - start_time
    app.decision=app.decision.sort_index()
    app.decision.to_csv('resultat.csv')
    
    poids-=app.poids_restant
    return  temps,app.borneInf,poids


def bbsanselagage(request,poids,filename):
    fs = FileSystemStorage()
    app=Sac_a_dos_app2(poids,'.'+fs.url(filename)) 
    
    start_time=perf_counter()           
    app.lancer()

    temps=  perf_counter() - start_time
    print(app.decision)
    print("gain engendré: ",app.borneInf)
    #
    print("Nombre des noeuds non feuille visités : ", Sac_a_dos_app2.cpt)
    
    #celui la doit etre affecteeeeeee
    poids-=app.poids_restant
    return  temps,app.borneInf,poids

def progdyn(request,poids,filename):
    fs = FileSystemStorage()
    
    start_time=perf_counter()           
    gain, poids = SacADos3(poids,'.'+fs.url(filename))
    
    temps=  perf_counter() - start_time

  
    return  temps,gain,poids



def ica(request,poids,filename):
    fs = FileSystemStorage()
    
    app=ICA(poids,'.'+fs.url(filename),number_of_countries=100,percentage_of_imperialists=0.2,percentage_of_assimilation=0.5,percentage_of_assimilated_colonies=0.5,percentage_of_revolution=0.2,percentage_of_evolved_colonies=0.5,battle_cycles=20,stagnation_factor=None)
        
    start_time=perf_counter()           
    app.ica()
    temps=  perf_counter() - start_time
    gain=app.previous_best_benefit
    poids=app.weight
    return  temps,gain,poids


def recuitsimule(request,poids,filename):
    fs = FileSystemStorage()
    app = metaheuristique_Voisiange(poids,'.'+fs.url(filename),False,0.9,900)            
    start = perf_counter()
    app.recuit_simule()
    temps=  perf_counter() - start
    gain=app.f_s_courant
    poids=app.poids_final

    return  temps,gain,app.poids_final

def fireworks(request,poids,filename):
    fs = FileSystemStorage()

    app=FireWorks(poids,'.'+fs.url(filename),5,10,10,20,3)


    print("--- seconds ---")
    start_time = time.time()

    app.algofireworks()
    temps=  (time.time() - start_time) 

    print("--- %s seconds ---" % (time.time() - start_time))

    maxim=app.getbestsol()
    gain=app.f(app.solutions[maxim])
   
    poids=np.dot(app.solutions[maxim],app.data["volumes"])
    return  temps,gain,poids

def triparval(request,poids,filename):
    fs = FileSystemStorage()
    start_time = perf_counter()
    maxValue, iRes, cap = TriParValeur.ValeurMax('.'+fs.url(filename), poids)
    temps=  perf_counter() - start_time
    print("Gains, Volumes")
    for obj in iRes:
        print(obj.val, obj.p, sep=', ')
    print("La valeur maximum du sac à dos =", maxValue)
    print("capacite = ", cap)
    print("Temps d’exécution : %s secondes " % (perf_counter() - start_time))
          
    return  temps,maxValue,cap

def tripardensite(request,poids,filename):
    fs = FileSystemStorage()
    start_time = perf_counter()
    maxValue, iRes, cap = TriParDensite.ValeurMax('.'+fs.url(filename), poids)
    temps=  perf_counter() - start_time

    print("Gains, Volumes")
    print("La valeur maximum du sac à dos =", maxValue)
    print("capacite = ", cap)
    print("Temps d’exécution : %s secondes " % (perf_counter() - start_time))
      
    return  temps,maxValue,cap


def triparpoids(request,poids,filename):
    fs = FileSystemStorage()        
    start_time=perf_counter()    
    maxValue, iRes, cap = TriParPoids.ValeurMax('.'+fs.url(filename), poids)
    temps=  perf_counter() - start_time
    print("Gains, Volumes")
    for obj in iRes:
        print(obj.val, obj.p, sep=', ')
    print("La valeur maximum du sac à dos =", maxValue)
    print("capacite = ", cap)
    print("Temps d’exécution : %s secondes " % (perf_counter() - start_time))
    
    
    return  temps,maxValue,cap

def Hybridel(request,poids,filename):
    fs = FileSystemStorage()
    app=LRH(poids,'.'+fs.url(filename),number_of_countries=100,percentage_of_imperialists=0.15,percentage_of_assimilation=0.5,percentage_of_assimilated_colonies=0.3,percentage_of_RC=0.3,percentage_of_evolved_colonies=0.8,battle_cycles=30, temperature= 900, alpha=0.9)
    start_time=perf_counter() 
    app.lrh()
    temps=  perf_counter() - start_time
    gain=app.previous_best_benefit
    poids=app.weight
    return  temps,gain,poids

def Hybrideh(request,poids,filename):
    fs = FileSystemStorage()
     
    app=ICA(poids,'.'+fs.url(filename),number_of_countries=100,percentage_of_imperialists=0.15,percentage_of_assimilation=0.5,percentage_of_assimilated_colonies=0.5,percentage_of_revolution=0.1,percentage_of_evolved_colonies=0.5,battle_cycles=2,stagnation_factor=None)

    start_time=perf_counter()           
    app.ica()
    appp=metaheuristique_Voisiange2(poids,'.'+fs.url(filename),app.countries[0].solution["decimal"], 0.9, 900)
    appp.recuit_simule()
    temps=  perf_counter() - start_time
    gain=appp.f_s_courant
      #celui la doit etre affecteeeeeee
    poids=appp.poid
    return  temps,gain,poids


def index1(request):
    
    return render(request, 'algoritmmanagement/index.html')