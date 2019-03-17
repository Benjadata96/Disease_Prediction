#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 14:06:38 2019

@author: benjaminsalem
"""
import numpy as np
import pandas as pd 

class EDA():
    
    def __init__(self, raw_table):
        
        self.raw_table = raw_table
        self.training_table = None
        print('..EDA initialized, ready for analysis and modifications..')
        
        
    def first_exploration(self): #Exploration des patients et cellules étudiés
        
        print('Cell type ratio : \n{}\n'.format(self.raw_table.cell_type.value_counts()))
        print('Patient state ratio : \n{}\n '.format(self.raw_table.patient_state.value_counts()))
        print('Nombre de patients unique : {}\n'.format(len(self.raw_table.patient_name.unique())))        
        print('Nombre de cellules unique : {}\n'.format(len(self.raw_table.cell_name.unique())))        
        print('Patient names : \n{}\n'.format(self.raw_table.patient_name.value_counts()))
        print('Cell names : \n{}'.format(self.raw_table.cell_name.value_counts()))
        
        
    def var_to_binary(self): # Conversion de l'état du patient et du type de cellule en variable binaire
        
        self.raw_table['cell_type'] = self.raw_table.cell_type.apply(lambda x: 1 if x=='B' else 0)
        self.raw_table['patient_state'] = self.raw_table.patient_state.apply(lambda x:1 if x=='malade' else 0)
        print("..Variables 'cell_type' and 'patient_state' are now integer binary..")
        
        print("Nombre de patients malades : {}".format(self.raw_table.groupby('patient_name').patient_state.sum().value_counts()[0]))

        
    def modify_cell_name(self): # Modification du nom des cellules en enlevant la partie initiale du nom sur le patient
        
        print('..Checking if all cell names starts with M1..')
        print('..Will print line number if not M1 type..')
        for cell_name in self.raw_table.cell_name.values: #on vérifie que le nom des cellules commence toujours par M1
            if 'M1_' not in cell_name :
                print('M1 not in cell name')
                print(cell_name)
                
        self.raw_table['cell_name'] = self.raw_table.cell_name.apply(lambda x: x[x.find('M1_'):]) 
        # Changement du nom des cellules : le patient n'intervient plus dans le nom 
        
        print('..Cell names modified..')
        print('Nombre de cellules unique après modification : {}\n'.format(len(self.raw_table.cell_name.unique())))
        print('Cell names : \n{}'.format(self.raw_table.cell_name.value_counts()))
        print("Nombre d'occurences moyen pour une cellule : {:.2f}".format(self.raw_table.cell_name.value_counts().mean()))
        print("Nombre d'occurences médiant pour une cellule : {:.2f}".format(self.raw_table.cell_name.value_counts().median()))
        
    
    def cell_name_to_dummy(self): # Conversion du nom des cellules en variables binaires
        
        cell_name = pd.get_dummies(self.raw_table.cell_name, dtype='int64')
        self.raw_table = self.raw_table.drop('cell_name',axis=1)
        self.raw_table['spectre']=self.raw_table.spectre.astype('int64')
        
        print(self.raw_table.info())
        
        self.training_table = pd.concat([self.raw_table, cell_name],axis=1)  
        
        
    def patient_name_indexing(self): # Reindéxation du dataframe en utilisant le nom du patient
        
        self.training_table = self.training_table.set_index('patient_name')
        print("..'patient_name' set as index of the dataframe..")
        print("..Training table ready..")
        
        return(self.training_table)
        
    def get_redundant_pairs(self,df): # Récupération de la diagonale et partie triangulaire basse de la matrice de corrélation
        # Dont on n'aura pas besoin pour raison de symétrie
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        
        return (pairs_to_drop)

    def get_top_abs_correlations(self,df, n):# Récupération des n variables les plus corrélées
        
        au_corr = df.corr().abs().unstack()
        labels_to_drop = self.get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        
        return (au_corr[0:n])
    

        
        
