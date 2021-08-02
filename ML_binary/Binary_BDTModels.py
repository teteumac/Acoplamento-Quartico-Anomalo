#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import mplhep as hep
from pandas import set_option
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import h5py
from sklearn.metrics import auc,make_scorer,fbeta_score,precision_score,recall_score,accuracy_score,log_loss,roc_auc_score,classification_report,f1_score,confusion_matrix,roc_curve,precision_recall_curve,average_precision_score
from joblib import dump, load
import catboost
from catboost import CatBoostClassifier
import argparse
import math
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser(description = 'Criando o modelo de treinamento')
parser.add_argument('--DataSet_Signal', help = 'DataSet contendo os eventos do sinal anomalo' )
parser.add_argument('--DataSet_Backgr', help = 'DataSet contendo os eventos de Background' )
parser.add_argument('--DataSet_SM', help = 'DataSet contendo os eventos do Modelo Padrao' )
parser.add_argument('--DataSet_Dados', help = 'DataSet contendo os eventos dos Dados ' )
parser.add_argument('--Label_Signal', help = 'Label do Acoplamento Anomalo ' )


args = parser.parse_args()

DataSet_Signal_ = args.DataSet_Signal
DataSet_SM_ = args.DataSet_SM
DataSet_Backgr_ = args.DataSet_Backgr
DataSet_Dados_ = args.DataSet_Dados
label_anomalo = args.Label_Signal

print( 'DataSet Signal Anomalo'+label_anomalo, DataSet_Signal_ )


def do_scaler(X,Y, scaler_X, scaler_Y):
    if scaler_X == None:
        X_scaled = X
    else:
        X_scaled = scaler_X.fit_transform(X)
    if scaler_Y == None:
        Y_scaled = Y
    else:
        Y_scaled = scaler_Y.fit_transform(Y)
    return X_scaled, Y_scaled, scaler_X, scaler_Y

def test_model(X,Y, model, scaler_X = None, verbose = 1):
    if scaler_X is not None:
        X_scaled = scaler_X.transform(X)
    else:
        X_scaled = X
    y_probs = model.predict_proba(X_scaled)  # calculate the probability
    preds = model.predict(X_scaled)
    prec, rec, thresh = precision_recall_curve(Y, y_probs[:,1])
    bidx = np.argmax(prec * rec)
    best_cut = thresh[bidx]
    print(best_cut)

    preds = y_probs[:,1] >= best_cut

    if verbose == 1:

        fpr, tpr, thresholds = metrics.roc_curve(Y, y_probs[:,1], drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')
        print("Purity in test sample     : {:2.2f}%".format(100 * precision_score(Y, preds)))
        print("Efficiency in test sample : {:2.2f}%".format(100 * recall_score(Y, preds)))
        print("Accuracy in test sample   : {:2.2f}%".format(100 * accuracy_score(Y, preds)))


PATH = '/eos/home-m/matheus/SWAN_projects/Acoplamento_Quartico_Anomalo/'
plt.style.use(hep.style.ROOT)


def open_file_MC( file ):
    df = None
    with h5py.File( file , 'r' ) as f:
        dset = f[ 'dados' ]
        array = np.array( dset ) 
        array_cut = (array[:,0] > 600) #& (array[:,1] > 200) & (array[:,2] > 2) & (array[:,3] > 2) & (array[:,4] > 200) & (array[:,5] < 2.4) & (array[:,7] < 0.6) & (array[:,8] > 40) & (array[:,9] > 53)  & (array[:,10] < 2.4)
        DataSet_ = array[array_cut]        
        dataframe = pd.DataFrame( DataSet_ , columns =  ['Mww', 'Pt_W_lep', 'Acoplanaridade_Whad_Wlep', 'Acoplanaridade_jatos_MET' ,
'jetAK8_pt','jetAK8_eta', 'jetAK8_prunedMass','jetAK8_tau21', 'METPt', 'muon_pt', 'muon_eta', 'ExtraTracks',
'Btag', 'Mx', 'Mww/Mx','Norm','weight'] )
        return dataframe 

def open_file_signal( file ):
    df = None
    with h5py.File( file , 'r' ) as f:
        dset = f[ 'dados' ]
        array = np.array( dset ) 
        arrau_cut_xi =  (array[:,0] > 600) & (array[:,15] > 0.04) & (array[:,16] > 0.04) & (array[:,15] < 0.111) & (array[:,16] < 0.138)
        DataSet_ = array[arrau_cut_xi]
        #DataSet_ = array
        Mx = 13000 * ( np.sqrt( DataSet_[:,15] * DataSet_[:,16] ) )
        Yx = 0.5 * ( np.log( DataSet_[:,16] / DataSet_[:,15] ) )
        Mww_Mxx =  DataSet_[:,0] / Mx
        Yww_Yx = DataSet_[:,14] - Yx
        DataSet = np.concatenate( ( DataSet_, Mx.reshape(-1,1), Yx.reshape(-1,1), Mww_Mxx.reshape(-1,1), Yww_Yx.reshape(-1,1) ), axis = 1 )
        dataframe = pd.DataFrame(DataSet,columns=['Mww', 'Pt_W_lep', 'dPhi_Whad_Wlep', 'dPhi_jatos_MET', 'jetAK8_pt','jetAK8_eta', 'jetAK8_prunedMass',
'jetAK8_tau21', 'METPt', 'muon_pt', 'muon_eta', 'ExtraTracks', 'PUWeight', 'Yww', 'Btag', 'xi1', 'xi2','weight','Mx','Yx','Mww/Mx','Yww_Yx'])
        dataframe['Acoplanaridade_Whad_Wlep'] = 1 - dataframe['dPhi_Whad_Wlep'].abs()/math.pi
        dataframe['Acoplanaridade_jatos_MET'] = 1 - dataframe['dPhi_jatos_MET'].abs()/math.pi
        return dataframe 

def open_file_datadriven( file ):
    df = None
    with h5py.File( file , 'r' ) as f:
        dset = f[ 'dados' ]
        array = np.array( dset ) 
        array_cut = (array[:,0] > 500) #& (array[:,1] >= 200) & (array[:,2] >= 2) & (array[:,3] >= 2) & (array[:,4] >= 200) & (array[:,5] <= 2.4) & (array[:,7] <=0.6) & (array[:,8] >= 40) & (array[:,9] >= 53)  & (array[:,10] <= 2.4)
        DataSet_ = array[array_cut]        
        dataframe = pd.DataFrame( DataSet_ , columns = ['Mww','Pt_W_lep','dPhi_Whad_Wlep','dPhi_jatos_MET','jetAK8_pt','jetAK8_eta','jetAK8_prunedMass','jetAK8_tau21','METPt','muon_pt','muon_eta','ExtraTracks', 'Yww','Btag','xi1','xi2', 'Mx','Yx', 'Mww/Mx','Yww_Yx', 'weight' ] )
        dataframe['Acoplanaridade_Whad_Wlep'] = 1 - dataframe['dPhi_Whad_Wlep'].abs()/math.pi
        dataframe['Acoplanaridade_jatos_MET'] = 1 - dataframe['dPhi_jatos_MET'].abs()/math.pi       
        return dataframe

def open_file_Data( file ):
    df = None
    with h5py.File( file , 'r' ) as f:
        dset = f[ 'dados' ]
        array = np.array( dset ) 
        array_cut =   (array[:,0] > 600) #& (array[:,1] > 200) & (array[:,2] > 2) & (array[:,3] > 2) & (array[:,4] > 200) & (array[:,5] < 2.4) & (array[:,7] < 0.6) & (array[:,8] > 40) & (array[:,9] > 53)  & (array[:,10] < 2.4)
        DataSet_ = array[array_cut]    
        Mx = 13000 * ( np.sqrt( DataSet_[:,15] * DataSet_[:,16] ) )
        Yx = 0.5 * ( np.log( DataSet_[:,16] / DataSet_[:,15] ) )
        Mww_Mxx =  DataSet_[:,0] / Mx
        Yww_Yx = DataSet_[:,14] - Yx
        DataSet = np.concatenate( ( DataSet_, Mx.reshape(-1,1), Yx.reshape(-1,1), Mww_Mxx.reshape(-1,1), Yww_Yx.reshape(-1,1) ), axis = 1 )        
        dataframe = pd.DataFrame( DataSet , columns = ['Mww', 'Pt_W_lep', 'dPhi_Whad_Wlep', 'dPhi_jatos_MET', 'jetAK8_pt','jetAK8_eta', 
'jetAK8_prunedMass','jetAK8_tau21', 'METPt', 'muon_pt', 'muon_eta', 'ExtraTracks', 'PUWeight', 'Btag',
'Yww', 'xi1', 'xi2','arm1', 'arm2', 'ismultirp_1','ismultirp_2', 'Mx','Yx','Mww/Mx','Yww_Yx']  )
        dataframe['Acoplanaridade_Whad_Wlep'] = 1 - dataframe['dPhi_Whad_Wlep'].abs()/math.pi
        dataframe['Acoplanaridade_jatos_MET'] = 1 - dataframe['dPhi_jatos_MET'].abs()/math.pi
        MultiRP = ( dataframe['ismultirp_1'] == 1 ) & ( dataframe['ismultirp_2'] == 1 ) 
        dataframe = dataframe[MultiRP]  
        return dataframe



select_columns = ['Mww', 'Pt_W_lep', 'Acoplanaridade_Whad_Wlep', 'Acoplanaridade_jatos_MET' ,'jetAK8_pt','jetAK8_eta', 'jetAK8_prunedMass','jetAK8_tau21', 'METPt', 'muon_pt', 'muon_eta', 'ExtraTracks','Btag', 'Mx', 'Mww/Mx']

SM      = open_file_signal( PATH + DataSet_SM_ )
ANOMALO = open_file_signal( PATH + DataSet_Signal_ )

label_signal  = pd.DataFrame( [1]*len( ANOMALO ) )
label_signalSM = pd.DataFrame( [1]*len( SM ) )

SM = pd.concat( [ SM, label_signalSM ], axis = 1 ).rename(columns={0: 'label'})
print('Shape for Standard Model --> ', SM.shape)

ANOMALO = pd.concat( [ ANOMALO, label_signal ], axis = 1 ).rename(columns={0: 'label'})
print('Shape for Anomalo  --> ', ANOMALO.shape)

data_set_back_multirp = open_file_MC( PATH + DataSet_Backgr_ )

#data_set_back_multirp = open_file_datadriven( PATH + DataSet_Backgr_ )
#data_set_back_multirp = pd.DataFrame( data_set_back_multirp , columns = ['Mww','Pt_W_lep','dPhi_Whad_Wlep','dPhi_jatos_MET','jetAK8_pt','jetAK8_eta','jetAK8_prunedMass','jetAK8_tau21','METPt','muon_pt','muon_eta','ExtraTracks', 'Yww','xi1','xi2', 'Mx','Yx', 'Mww/Mx','Yww_Yx', 'weight' ]  )


label_back = pd.DataFrame( [0]*len( data_set_back_multirp ) )
data_set_back_multirp = pd.concat( [ data_set_back_multirp, label_back ], axis = 1 ).rename(columns={0: 'label'})
print('Shape do Background', data_set_back_multirp.shape)

Dataset_Signal_Back  = pd.concat( [ ANOMALO , data_set_back_multirp, SM  ], axis = 0, sort = False )

data_set_dados_multirp = open_file_Data( PATH + DataSet_Dados_ )

print('Shape dos Dados', data_set_dados_multirp.shape)

from sklearn.model_selection import train_test_split

test_size = 0.40
DataSet_Train_, DataSet_Test_ = train_test_split( Dataset_Signal_Back, test_size = test_size, random_state = 42, stratify = Dataset_Signal_Back.label )

#DataSet_Train_, DataSet_Validation_ = train_test_split( DataSet_Train_, test_size = 20, random_state = 42, stratify = Dataset_Signal_Back.label )

y_train = DataSet_Train_['label']
y_test  = DataSet_Test_['label']
#y_valid = DataSet_Validation_['label']

DataSet_Test_weight_signal = DataSet_Test_[DataSet_Test_['label']==1]['weight']
DataSet_Test_weight_backgr = DataSet_Test_[DataSet_Test_['label']==0]['weight']

DataSet_Train_weight_signal = DataSet_Train_[DataSet_Train_['label']==1]['weight']
DataSet_Train_weight_backgr = DataSet_Train_[DataSet_Train_['label']==0]['weight']



DataSet_Train = DataSet_Train_[select_columns]
#DataSet_Validation = DataSet_Validation_[select_columns]
DataSet_Test  = DataSet_Test_[select_columns]
data_set_dados_multirp_ = data_set_dados_multirp[select_columns]
 
scaler = StandardScaler()
X_train_norm = scaler.fit_transform( DataSet_Train )
#X_vali_norm = scaler.fit_transform( DataSet_Validation )
X_test_norm = scaler.transform( DataSet_Test )
X_dados_norm = scaler.transform( data_set_dados_multirp_ )

Y_train_norm = y_train
Y_test_norm = y_test

model_number = 4
# Create Model
if model_number == 1:
    model_name = "RF"
    # Random Forest / Best Result - AUC = 0.8663
    param_search = {
        'RF__n_estimators': [100, 200, 300, 400],
        'RF__max_depth': list(range(4,8))

    }
    # create pipeline
    estimators = []
    estimators.append(('RF', RandomForestClassifier()))
    model = Pipeline(estimators)
elif model_number == 2:
    model_name = "ADAB"
    param_search = {"ADAB__base_estimator__max_depth": [1, 2, 3, 4, 5],
                  "ADAB__n_estimators": [700, 800, 900]
                  }

    DTC = DecisionTreeClassifier(max_features="auto", class_weight="balanced", max_depth=None)

    # create pipeline
    estimators = []
    estimators.append(('ADAB', AdaBoostClassifier(base_estimator=DTC)))
    model = Pipeline(estimators)
elif model_number == 3:
    model_name = "XGB"
    param_search = {'XGB__max_depth': [2, 3, 4, 5],
                  'XGB__n_estimators': [200, 400, 600, 800],
                  'XGB__learning_rate': [0.15, 0.2],
                  'XGB__gamma': [0, 0.1]
                  }

    # create pipeline
    estimators = []
    #estimators.append(('standardize', StandardScaler()))
    estimators.append(('XGB', XGBClassifier( booster='gbtree')))
    model = Pipeline(estimators)
elif model_number == 4:
    # Lightgbm
    model_name = "LGBM"
    param_search = {'LGB__num_leaves': [ 24 ],
                  'LGB__learning_rate': [ 0.01 ],
                  'LGB__n_estimators': [  2000 ],
                  'LGB__min_child_samples': [ 50, 100, 500, 900 ],
                  'LGB__subsample': [0.2, 0.3,0.4,0.5],
                  'LGB__colsample_bytree': [0.2,0.3,0.4,0.5],
                  'LGB__boosting_type': ['gbdt'],
		  'LGB__reg_alpha':[1,2,3,4,5],
		  'LGB__reg_lambda':[1,2,3,4,5],	
                  #'LGB__max_depth': [2, 6, 10] #ou coloca ele ou coloca LGB__min_child_samples
                  }

    # create pipeline
    estimators = []
    #estimators.append(('standardize', StandardScaler()))
    estimators.append(('LGB', LGBMClassifier( random_state = 42, subsample_freq=1 )))
    model = Pipeline(estimators)

elif model_number == 5:
    # Lightgbm
    model_name = "CatBoost"
    param_search = {'CatBoost__depth': [ 3, 9, 6 ], # Profundidade da árvore. Pode ser qualquer número inteiro até 32. Valores bons no intervalo de 1 a 10
                    'CatBoost__iterations': [ 500, 1000, 2000 ], # O número máximo de árvores que podem ser construídas ao resolver problemas de aprendizado de máquina.
                    'CatBoost__learning_rate': [0.03,0.0005, 0.8], # usado para reduzir a etapa de gradiente. Isso afeta o tempo geral de treinamento: quanto menor o valor, mais iterações são necessárias para o treinamento.
                    'CatBoost__l2_leaf_reg': [ 3, 8, 18, 30], # tente valores diferentes para o regularizador para encontrar o melhor possível. Quaisquer valores positivos são permitidos.
                    'CatBoost__border_count': [ 100, 200, 300 ], # Para classificação de 2 classes, use 'LogLoss' ou 'CrossEntropy'. Para multiclasse, use 'MultiClass'.
                 }
            


    # create pipeline
    estimators = []
    #estimators.append(('standardize', StandardScaler()))
    estimators.append(('CatBoost', CatBoostClassifier( loss_function = "CrossEntropy" )))
    model = Pipeline(estimators)

from joblib import dump

def make_model(X_train_norm,Y_train_norm,param_search,filename):
    n_iter = 8
    cv = 3
    #scoring = make_scorer(fbeta_score, beta=0.5)
    #scoring = 'precision'
    scoring = 'f1'
    #scoring = None

    #search object

    search = GridSearchCV( estimator=model, param_grid=param_search, scoring = scoring, cv=cv, verbose=2, n_jobs=-1 )

    start_time = time.time()

    search_result = search.fit(X_train_norm, Y_train_norm )
    print("--- %s seconds ---" % (time.time() - start_time))
    id_ = time.strftime("%Y_%m_%d-%H_%M_%S")
    fileName_ = "LGBM_clf_{}_{}.joblib".format( label_anomalo, id_ )
    print ( "Saving model to {}".format( fileName_ ) )
    dump( search_result, fileName_ )
    # Results
    print("Model: %s Best: %f using %s" % (model_name, search_result.best_score_, search_result.best_params_))
    means = search_result.cv_results_['mean_test_score']
    stds = search_result.cv_results_['std_test_score']
    params = search_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return search_result   


import time 

def test_model(X,Y, model, scaler_X = None, verbose = 2):
    if scaler_X is not None:
        X_scaled = scaler_X.transform(X)
    else:
        X_scaled = X
    y_probs = model.predict_proba(X_scaled)  # calculate the probability
    preds = model.predict(X_scaled)
    prec, rec, thresh = precision_recall_curve(Y, y_probs[:,1])
    bidx = np.argmax(prec * rec)
    best_cut = thresh[bidx]
    print(best_cut)

    preds = y_probs[:,1] >= best_cut

    if verbose == 1:

        fpr, tpr, thresholds = roc_curve(Y, y_probs[:,1], drop_intermediate=False)
        print(f'AUC = {auc(fpr, tpr)}')
        print("Purity in test sample     : {:2.2f}%".format(100 * precision_score(Y, preds)))
        print("Efficiency in test sample : {:2.2f}%".format(100 * recall_score(Y, preds)))
        print("Accuracy in test sample   : {:2.2f}%".format(100 * accuracy_score(Y, preds)))
    return y_probs[:,1],best_cut


def Volta_Lula( DataSet_Test, y_test, DataSet_Train, y_train, DataSet_Test_weight_signal, DataSet_Test_weight_backgr, label_anomalo, label_save_model ):
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform( DataSet_Train )
    X_scaled_test = scaler.transform( DataSet_Test )
    X_dados_norm = scaler.transform( data_set_dados_multirp_ )

    n_iter = 8
    cv = 3
    #scoring = make_scorer(fbeta_score, beta=0.5)
    #scoring = 'roc_auc'
    scoring = 'f1'
    #scoring = None
    time_s_ = time.time()
    #search object

    search = GridSearchCV( estimator = model, param_grid = param_search, scoring = scoring ,cv = cv, verbose = 2 )

    start_time = time.time()

    search_result = search.fit( X_scaled_train, y_train )
    print("--- %s seconds ---" % ( time.time() - start_time ) )
    dump( search_result, label_save_model+'.joblib' )
    # Results
    print( "Model: %s Best: %f using %s" % ( model_name, search_result.best_score_, search_result.best_params_ ) )
    means = search_result.cv_results_[ 'mean_test_score' ]
    stds = search_result.cv_results_[ 'std_test_score' ]
    params = search_result.cv_results_[ 'params' ]
    time_e_ = time.time()
    print ( "Total time elapsed: {:.0f}".format( time_e_ - time_s_ ) )
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    fileName_ = "LGBM_clf_MonteCarlo_{}.joblib".format( label_anomalo )
    print ( "Saving model to {}".format( fileName_ ) )
    dump( search_result, fileName_ )
    
    print('\n\n ---------------- Result of traning for' + label_anomalo + ' with metrics classifeir---------------- \n' )

    y_probs_test = search_result.predict_proba( X_scaled_test )[:,1]  # calculate the probability
    prec_test, rec_test, thresh_test = precision_recall_curve(y_test, y_probs_test)
    f1_score_test = 2 * ( prec_test * rec_test ) / ( rec_test + prec_test )
    #bidx_test = np.argmax( prec_test * rec_test )
    bidx_test = np.argmax( f1_score_test )
    best_cut_test = thresh_test[bidx_test]

    preds_test = y_probs_test >= best_cut_test

    y_probs_train = search_result.predict_proba(X_scaled_train)[:,1]  # calculate the probability
    prec_train, rec_train, thresh_train = precision_recall_curve(y_train, y_probs_train)
    f1_score_train = 2 * ( prec_train * rec_train ) / ( rec_train + prec_train )
    #bidx_train = np.argmax( prec_train * rec_train )
    bidx_train = np.argmax( f1_score_train )
    best_cut_train = thresh_train[bidx_train]
    

    preds_train = y_probs_train >= best_cut_train


    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_probs_test, drop_intermediate=False)
    print('best_cut_test-->', best_cut_test)
    print(f'AUC = {auc(fpr_test, tpr_test)}')
    print("Purity in test sample     : {:2.2f}%".format(100 * precision_score(y_test, preds_test)))
    print("Efficiency in test sample : {:2.2f}%".format(100 * recall_score(y_test, preds_test)))
    print("Accuracy in test sample   : {:2.2f}%".format(100 * accuracy_score(y_test, preds_test)))
    print( "F1_score in test sample  : {:2.2f}%".format(100 * f1_score(y_test, preds_test)))
    print("\n")
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_probs_train, drop_intermediate=False)
    print('best_cut_train-->', best_cut_train)
    print(f'AUC = {auc(fpr_train, tpr_train)}')
    print("Purity in train sample     : {:2.2f}%".format(100 * precision_score(y_train, preds_train)))
    print("Efficiency in train sample : {:2.2f}%".format(100 * recall_score(y_train, preds_train)))
    print("Accuracy in train sample   : {:2.2f}%".format(100 * accuracy_score(y_train, preds_train)))
    print( "F1_score in train sample  : {:2.2f}%".format(100 * f1_score(y_train, preds_train)))        

    
    print('\n\n -------------------- Baseline using Logistic Regressor -------------------- \n\n ')

    Regre = LogisticRegression(max_iter=1000)
    Regre.fit( X_scaled_train, y_train )     

    y_probs_train = Regre.predict_proba( X_scaled_train )[:,1] 
    prec_train, rec_train, thresh_train = precision_recall_curve(y_train, y_probs_train)
    #bidx_train = np.argmax(prec_train * rec_train)
    bidx_train = np.argmax(2* prec_train * rec_train / ( prec_train + rec_train ) )

    best_cut_train = thresh_train[bidx_train]
    print( 'Best Cut of train for baseline-->', best_cut_train )

    preds__train = y_probs_train >= best_cut_train

    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_probs_train, drop_intermediate=False)
    print(f'AUC = {auc(fpr_train, tpr_train)}')
    print("Purity in train sample for baseline     : {:2.2f}%".format(100 * precision_score(y_train, preds__train)))
    print("Efficiency in train sample for baseline : {:2.2f}%".format(100 * recall_score(y_train, preds__train)))
    print("Accuracy in train sample   for baseline : {:2.2f}%".format(100 * accuracy_score(y_train, preds__train)))
    print("F1_score in train sample   for baseline : {:2.2f}%".format(100 * f1_score(y_train, preds__train)))

    y_probs_train = Regre.predict_proba( X_scaled_test )[:,1]
    prec_train, rec_train, thresh_train = precision_recall_curve(y_test, y_probs_test)
    #bidx_train = np.argmax(prec_train * rec_train)
    bidx_train = np.argmax(2* prec_train * rec_train / ( prec_train + rec_train ) )

    best_cut_train = thresh_train[bidx_train]
    print( '\nBest Cut of test for baseline-->', best_cut_train )

    preds__train = y_probs_train >= best_cut_train

    fpr_train, tpr_train, thresholds_train = roc_curve(y_test, y_probs_train, drop_intermediate=False)
    print(f'AUC = {auc(fpr_train, tpr_train)}')
    print("Purity in test sample for baseline     : {:2.2f}%".format(100 * precision_score(y_test, preds__train)))
    print("Efficiency in test sample for baseline : {:2.2f}%".format(100 * recall_score(y_test, preds__train)))
    print("Accuracy in test sample   for baseline : {:2.2f}%".format(100 * accuracy_score(y_test, preds__train)))
    print("F1_score in test sample   for baseline : {:2.2f}%".format(100 * f1_score(y_test, preds__train)))


Volta_Lula(  DataSet_Test, y_test, DataSet_Train, y_train, DataSet_Test_weight_signal, DataSet_Test_weight_backgr, label_anomalo, label_anomalo )

'''
search_result = make_model(X_train_norm, Y_train_norm,param_search,label_anomalo)
y_probs,best_cut = test_model(X_test_norm,y_test, search_result, scaler, 1)
predict_dados = search_result.predict_proba( data_set_dados_multirp[ select_columns ])[:,1]
y_probs_train ,best_cut_train = test_model(X_train_norm,y_train, search_result, scaler, 1)

n_events_back_after_cut = y_probs[ y_test == 0 ] > best_cut
n_events_signal_after_cut = y_probs[ y_test == 1 ] > best_cut
n_eventos_Data_after_cut = predict_dados[ predict_dados >= best_cut ]

print(' --------------- Anomalo'+label_anomalo+' --------------- ')
print('Numero de eventos de background depois do corte -->', DataSet_Test_weight_backgr[ n_events_back_after_cut ].sum() / test_size )
print('Numero de eventos de signal depois do corte -->', DataSet_Test_weight_signal[ n_events_signal_after_cut].sum() / test_size )
print('Numero de eventos de dados depois do corte -->', len( n_eventos_Data_after_cut ) )

fpr_test, tpr_test, thresholds_test = roc_curve(Y_test, y_probs, drop_intermediate=False)
print('best_cut_test-->', best_cut)
print(f'AUC = {auc(fpr_test, tpr_test)}')
print("Purity in test sample     : {:2.2f}%".format(100 * precision_score(Y_test_norm, y_probs)))
print("Efficiency in test sample : {:2.2f}%".format(100 * recall_score(Y_test_norm, y_probs)))
print("Accuracy in test sample   : {:2.2f}%".format(100 * accuracy_score(Y_test1_norm, y_probs)))
print( "F1_score in test sample  : {:2.2f}%".format(100 * f1_score(Y_test_norm, y_probs)))
print("\n")
fpr_train, tpr_train, thresholds_train = roc_curve(Y_train_norm, y_probs_train, drop_intermediate=False)
print('best_cut_train-->', best_cut_train)
print(f'AUC = {auc(fpr_train, tpr_train)}')
print("Purity in train sample     : {:2.2f}%".format(100 * precision_score(Y_train_norm, y_probs_train)))
print("Efficiency in train sample : {:2.2f}%".format(100 * recall_score(Y_train_norm, y_probs_train)))
print("Accuracy in train sample   : {:2.2f}%".format(100 * accuracy_score(Y_train_norm, y_probs_train)))
print( "F1_score in train sample  : {:2.2f}%".format(100 * f1_score(Y_train_norm, y_probs_train)))
'''
