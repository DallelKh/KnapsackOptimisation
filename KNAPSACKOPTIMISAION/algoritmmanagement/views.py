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
                poids_restant=Sac_a_dos_app.poids-self.arbre.cout_branche(noeud_pere)
                nbr_exemplaires_max=int(poids_restant/Sac_a_dos_app.donnees.loc[num_obj,'volumes'])
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


def upload(request):
    ##getting file
    for key, uploaded_file in request.FILES.items():
        if request.method == 'POST' :
            
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
            poids=0
            bbavecelagage(request,130,filename)
            return render(request, 'algoritmmanagement/upload.html', {
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
    print("Temps d execution : %s secondes " % (perf_counter() - start_time))
    app.decision=app.decision.sort_index()
    app.decision.to_csv('resultat.csv')
    print("gain engendré: ",app.borneInf,'\n\n')
    pass 



def index1(request):
    
    return render(request, 'algoritmmanagement/index.html')