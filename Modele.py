#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:45:17 2019

@author: benjaminsalem
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score
import warnings

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 100
import matplotlib.pyplot as plt


class Model():

    def __init__(self, name, train_test_split, test_val_split, cross_valid_fold, is_malade, is_type_cell):
        
        self.tt_split = train_test_split # ratio du test_set
        self.tv_split = test_val_split # ratio du validation set
        self.cv_fold = cross_valid_fold # nombre de plis pour la validation croisée
        self.is_malade = is_malade # prédit-on l'état du patient ou le nombre de cellules B 
        self.is_type_cell = is_type_cell # prend-on en considération les cellules B dans la prédiction de l'état du patient
        self.name = name # nom du modèle dans {'Log_Reg','Gradient_Boosting'}
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.Y_train = None
        self.Y_val = None
        self.Y_test = None
        self.patient_state_test = None
        self.best_15_features = []
        
        print('..Model '+str(self.name)+' initialized..')
        print('..Test set will be a '+ str(self.tt_split) +' ratio of the dataset..')
        print('..There will be '+ str(self.cv_fold) +' folds in the cross-validation..')
        
        if self.is_malade == 1:
            print('..This model will predict the patient state..')
        else:
            print('..This model will predict the cell type (B or TNK)..')
        
        
    def prepare_inputs_outputs(self,df): # Implémentatin de X_train, X_val, X_test, Y_train, Y_val, Y_test
        
        if self.is_malade == 0: # prediction du type de la cellule
            outputs = df.cell_type.values
        else: # prediction de l'état du patient
            outputs = df.patient_state.values
            
            self.patient_state_test = df.groupby('patient_name').patient_state.sum().apply(lambda x: 1 if x>=1 else 0)
            self.patient_state_test = self.patient_state_test.rename("is_malade_true")
        
        if self.is_type_cell == 1 : # on ne prend pas en compte l'état du patient
            inputs = df.drop('patient_state',axis=1)
        else: # on ne prend pas en compte l'état du patient ni le type de la cellule
            inputs = df.drop(['cell_type','patient_state'],axis=1)
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(inputs, outputs, random_state = 0, test_size =self.tt_split) 
        self.X_test, self.X_val, self.Y_test, self.Y_val = train_test_split(self.X_test, self.Y_test, random_state = 0, test_size =self.tv_split) 
        
        print('Shape de X_train : {}'.format(self.X_train.shape))
        print('Shape de X_val : {}'.format(self.X_val.shape))
        print('Shape de X_test : {}'.format(self.X_test.shape))
        print('Shape de Y_train : {}'.format(self.Y_train.shape))
        print('Shape de Y_val : {}'.format(self.Y_val.shape))
        print('Shape de X_train : {}'.format(self.Y_test.shape))

        print('..Inputs & outputs computed..')
        
        
    def training_model(self,param_grid):
        warnings.filterwarnings("ignore") # permet d'ignorer les dépréciations
        
        if self.name == 'Log_Reg': # initialisation du modèle de régression logistique
            grid = GridSearchCV(LogisticRegression(random_state=0),param_grid = param_grid, cv=self.cv_fold)
            print('..'+str(self.name)+' model initialized..')
        if self.name == 'Gradient_Boosting': # initialisation du modèle de gradient boosting
            grid = GridSearchCV(GBC(random_state=0),param_grid = param_grid, cv=self.cv_fold)
            print('..'+str(self.name)+' model initialized..')
        
        grid.fit(self.X_train, self.Y_train)
        self.best_model = grid.best_estimator_
        
        print('..Model is trained..')
        print("Best cross-validation score: {:.2f}".format(grid.best_score_))
        print("Best hyperparameters: ", grid.best_params_)
        print("Best model : ", grid.best_estimator_)
        print("Test set score: {:.2f}".format(grid.score(self.X_test, self.Y_test)))

        
    def F1_Score_Precision_Recall(self): # calcul des metrics F1 score, recall, precision
        
        print('Valeur du F1 score sur le validation set : {:.2f}'.format(f1_score(self.Y_val, self.best_model.predict(self.X_val))))
        print('Valeur du recall sur le validation set : {:.2f}'.format(recall_score(self.Y_val, self.best_model.predict(self.X_val))))
        print('Valeur de la précision sur le validation set : {:.2f}'.format(precision_score(self.Y_val, self.best_model.predict(self.X_val))))
        
    
    def plotting_ROC(self): # plot la courbe ROC et precision-recall
        
        fpr,tpr,thresholds = roc_curve(self.Y_val, self.best_model.decision_function(self.X_val))
        plt.figure()
        plt.plot(fpr,tpr,label = 'ROC Curve')
        plt.xlabel("FPR")
        plt.ylabel("TPR (recall)")
        plt.legend(loc='best')
        plt.show()
        
        print('Area Under the Curve : {}'.format(roc_auc_score(self.Y_val, self.best_model.decision_function(self.X_val))))

        precision, recall, threshold = precision_recall_curve(self.Y_val, self.best_model.decision_function(self.X_val))
        plt.plot(precision, recall, label = 'Precision-Recall Curve')
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.legend(loc='best')
        plt.show()
        
        
    def plot_feature_importances(self): # plot l'importance des features dans le modèle
        
        print('..Plotting features importance in the model..')
        n_init_features = self.X_train.shape[1]
        n_features = 15
        
        if self.name == 'Log_Reg': # attribut 'coef_' du modèle  pour la régression logistique
            coefs_array = np.reshape(self.best_model.coef_, (n_init_features))
        if self.name == 'Gradient_Boosting':# attribut 'feature_importances' du modèle  pour le gradient boosting
            coefs_array = np.reshape(self.best_model.feature_importances_, (n_init_features))
        
        best_15_value = [] # on récupère les 15 features les plus importantes
        col = self.X_train.columns.tolist()
        for i in range(n_features):
            max_index = np.argmax(coefs_array)
            self.best_15_features.append(col[max_index])
            best_15_value.append(coefs_array[max_index])
            coefs_array = np.delete(coefs_array,max_index)

        plt.barh(np.arange(n_features), best_15_value, align='center')
        plt.xlabel("Feature coefficients")
        plt.yticks(np.arange(n_features), self.best_15_features) 
        plt.ylabel("Feature")
        plt.ylim(-1, n_features)
        plt.title("Coefficients of the 15 most important features in the model")
        plt.show()
        
    
    def final_pred(self, X, dataset_name): # calcul du nombre total de cellule B chez un patient 
                                           # ou calcul de la prédiction de l'état du patient (et non de la cellule)
        Y_pred = self.best_model.predict(X)
        X = X.reset_index()
        Y_pred_series = pd.Series(Y_pred,name='cell_type_pred')
        Y_pred_name = pd.concat([X['patient_name'],Y_pred_series],axis=1)
               
        if self.is_malade == 0 :
            if dataset_name == 'test_set':
                Y_pred_series = pd.Series(Y_pred,name='cell_type_pred')
                Y_pred_name = pd.concat([X['patient_name'],Y_pred_series],axis=1)
                final_result = Y_pred_name.groupby('patient_name').cell_type_pred.sum() #nombre de cellules B 
                print("Prédictions du nombre de cellules B pour chaque patient du "+dataset_name+" :\n {}".format(final_result))
            else:
                pass
        
        else:
            Y_pred_series = pd.Series(Y_pred,name='nb_cells_ill_patient')
            Y_pred_name = pd.concat([X['patient_name'],Y_pred_series],axis=1)
            Y_total_cells_per_patient = pd.Series(Y_pred_name.patient_name.value_counts(), name='total_cells')
            # nombre total de cellules étudiées pour chaque patient
            Y_total_cells_per_ill_patient = Y_pred_name.groupby('patient_name').nb_cells_ill_patient.sum()
            # nombre total de cellules prédites comme appartenant à un patient malade
            
            final_result = pd.concat([Y_total_cells_per_patient, Y_total_cells_per_ill_patient],axis=1)
            final_result['ratio_cells_ill_patient'] = np.round(final_result.nb_cells_ill_patient / final_result.total_cells,2)
            # pourcentage de cellules prédites comme appartenant à un patient malade
            final_result['is_malade_pred']=final_result.ratio_cells_ill_patient.apply(lambda x:1 if x>0.5 else 0)
            # HYPOTHESE : si le patient a plus de 50% de cellules qui sont prédites comme appartenant à un patient malade, alors il est malade
            final_result = final_result.join(self.patient_state_test)
            
            print("Nombre de patients étudiés dans le "+dataset_name+" : {}".format(len(final_result)))
            print("Nombre de patients prédits malades dans le "+dataset_name+" : {}".format(final_result.is_malade_pred.sum()))
            print("Nombre de patients réellement malades dans le "+dataset_name+" : {}".format(final_result.is_malade_true.sum()))
            print("F1-Score sur les prédictions de l'état des patients pour le "+dataset_name+"  : {:.2f}".format(f1_score(final_result.is_malade_true.values, final_result.is_malade_pred.values)))
            print("Recall sur les prédictions de l'état des patients pour le "+dataset_name+"  : {:.2f}".format(recall_score(final_result.is_malade_true.values, final_result.is_malade_pred.values)))
            print("Precision sur les prédictions de l'état des patients pour le "+dataset_name+"  : {:.2f}\n".format(precision_score(final_result.is_malade_true.values, final_result.is_malade_pred.values)))
  
  
    def main(self,df, param_grid):
        
        self.prepare_inputs_outputs(df)
        self.training_model(param_grid)
        self.F1_Score_Precision_Recall()
        self.plotting_ROC()
        self.plot_feature_importances()
        self.final_pred(self.X_val, "validation_set")
        self.final_pred(self.X_test, "test_set")
        
        return(self.best_model)






