#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import h5py
from joblib import dump, load
from sklearn.metrics import auc,make_scorer,fbeta_score,precision_score,recall_score,accuracy_score,log_loss,roc_auc_score,classification_report,f1_score,confusion_matrix,roc_curve,precision_recall_curve,average_precision_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

import argparse

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

PATH = '/eos/home-m/matheus/SWAN_projects/Acoplamento_Quartico_Anomalo/'

def open_file_MC( file ):
    df = None
    with h5py.File( file , 'r' ) as f:
        dset = f[ 'dados' ]
        array = np.array( dset )
        dataframe = pd.DataFrame( array , columns = ['Mww', 'Pt_W_lep', 'dPhi_Whad_Wlep', 'dPhi_jatos_MET', 'jetAK8_pt','jetAK8_eta', 'jetAK8_prunedMass',
'jetAK8_tau21', 'METPt', 'muon_pt', 'muon_eta', 'ExtraTracks', 'PUWeight', 'Yww', 'xi1', 'xi2', 'Mx', 'Yx',
'Mww/Mx', 'Yww_Yx', 'weight'] )
        return dataframe

def open_file_datadriven( file ):
    df = None
    with h5py.File( file , 'r' ) as f:
        dset = f[ 'dados' ]
        array = np.array( dset )
        dataframe = pd.DataFrame( array , columns = ['Mww','Pt_W_lep','dPhi_Whad_Wlep','dPhi_jatos_MET','jetAK8_pt','jetAK8_eta','jetAK8_prunedMass','jetAK8_tau21','METPt','muon_pt','muon_eta','ExtraTracks', 'Yww','xi1','xi2', 'Mx','Yx', 'Mww/Mx','Yww_Yx', 'weight' ]  )
        return dataframe

def open_file_DD( file ):
    df = None
    with h5py.File( file , 'r' ) as f:
        dset = f[ 'dados' ]
        array = np.array( dset )
        dataframe = pd.DataFrame( array , columns = ['Mww', 'Pt_W_lep', 'dPhi_Whad_Wlep', 'dPhi_jatos_MET', 'jetAK8_pt',
       'jetAK8_eta', 'jetAK8_prunedMass', 'jetAK8_tau21', 'METPt', 'muon_pt',
       'muon_eta', 'ExtraTracks', 'Yww', 'xi1', 'xi2', 'MultiRP1', 'MultiRP2',
       'Mx', 'Yx', 'Mww/Mx', 'Yww_Yx', 'weight'] )
        return dataframe

def open_file_Data( file ):
    df = None
    with h5py.File( file , 'r' ) as f:
        dset = f[ 'dados' ]
        array = np.array( dset )
        dataframe = pd.DataFrame( array , columns = ['Mww', 'Pt_W_lep', 'dPhi_Whad_Wlep', 'dPhi_jatos_MET', 'jetAK8_pt','jetAK8_eta', 'jetAK8_prunedMass',
'jetAK8_tau21', 'METPt', 'muon_pt', 'muon_eta', 'ExtraTracks', 'PUWeight', 'Yww', 'xi1', 'xi2','Mx', 'Yx',
'Mww/Mx', 'Yww_Yx'] )
        return dataframe

select_columns = ['Mww', 'Pt_W_lep', 'dPhi_Whad_Wlep', 'dPhi_jatos_MET', 'jetAK8_pt','jetAK8_eta', 'jetAK8_prunedMass', 'jetAK8_tau21', 'METPt', 'muon_pt', 'muon_eta', 'ExtraTracks', 'Yww', 'xi1', 'xi2', 'Mx', 'Yx', 'Mww/Mx', 'Yww_Yx']

SM      = open_file_MC( PATH + DataSet_SM_ )
ANOMALO = open_file_MC( PATH + DataSet_Signal_ )

label_signal  = pd.DataFrame( [1]*len( ANOMALO ) )
label_signalSM = pd.DataFrame( [0]*len( SM ) )

SM = pd.concat( [ SM, label_signalSM ], axis = 1 ).rename(columns={0: 'label'})
print('Shape for Standard Model --> ', SM.shape)

ANOMALO = pd.concat( [ ANOMALO, label_signal ], axis = 1 ).rename(columns={0: 'label'})
print('Shape for Anomalo  --> ', ANOMALO.shape)

#data_set_back_multirp = open_file_MC( PATH + DataSet_Backgr_ )
data_set_back_multirp = open_file_datadriven( PATH + DataSet_Backgr_ )


label_back = pd.DataFrame( [0]*len( data_set_back_multirp ) )
data_set_back_multirp = pd.concat( [ data_set_back_multirp, label_back ], axis = 1 ).rename(columns={0: 'label'})
print('Shape do Background', data_set_back_multirp.shape)

Dataset_Signal_Back  = pd.concat( [ ANOMALO , data_set_back_multirp, SM  ], axis = 0, sort = False )

data_set_dados_multirp = open_file_Data( PATH + 'DataSet_dados_multiRP.h5' )

print('Shape dos Dados', data_set_dados_multirp.shape)

from sklearn.model_selection import train_test_split

test_size = 0.35
DataSet_Train_, DataSet_Test_ = train_test_split( Dataset_Signal_Back, test_size = test_size, random_state = 42, stratify = Dataset_Signal_Back.label )

y_train = DataSet_Train_['label']
y_test  = DataSet_Test_['label']

DataSet_Test_weight_signal = DataSet_Test_[DataSet_Test_['label']==1]['weight']
DataSet_Test_weight_backgr = DataSet_Test_[DataSet_Test_['label']==0]['weight']

DataSet_Train_weight_signal = DataSet_Train_[DataSet_Train_['label']==1]['weight']
DataSet_Train_weight_backgr = DataSet_Train_[DataSet_Train_['label']==0]['weight']


DataSet_Train = DataSet_Train_[select_columns]
DataSet_Test  = DataSet_Test_[select_columns]


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from lightgbm import LGBMClassifier


grid_search = None
run_tables = False
train_model = True
run_grid_search = True
save_model = True
n_iter_search_ = 100

if train_model and run_grid_search:
    import time
    print( time.strftime("%Y/%m/%d %H:%M:%S", time.localtime() ) )
    time_s_ = time.time()

    from sklearn.model_selection import RandomizedSearchCV
    #from sklearn.model_selection import GridSearchCV
    #from scipy.stats import uniform

    param_distribs = {
        "base_estimator__max_depth": np.arange(2,9),
        "base_estimator__min_samples_split": np.arange(2,9),
        "n_estimators": 100 * np.arange(2,6),
        "learning_rate": 0.1 * np.arange(4,11)
        }
    #param_grid = [
    #    { "max_depth": np.arange(2,10),
    #      "n_estimators": 100 * np.arange(1,6),
    #      "learning_rate": 0.1 * np.arange(5,11) }
    #    ]

    grid_search = RandomizedSearchCV(
        AdaBoostClassifier(
            DecisionTreeClassifier(),
            algorithm="SAMME.R"
            ),
        param_distribs,
        n_iter=n_iter_search_, cv=2, verbose=1, n_jobs=-1, random_state=42
        )
    grid_search.fit( DataSet_Train, y_train )

    print ( grid_search.best_params_ )
    print ( grid_search.best_score_ )
    print ( grid_search.cv_results_ )

    time_e_ = time.time()
    print ( "Total time elapsed: {:.0f}".format( time_e_ - time_s_ ) )

# Build model

model_final = None

if train_model:
    if run_grid_search: 
        print ( grid_search.best_estimator_)
        model_final = grid_search.best_estimator_
    else:
        model_final = AdaBoostClassifier(
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_split=5
                ),
                n_estimators = 400,
                algorithm="SAMME.R",
                learning_rate = 0.4
                )
        model_final.fit( DataSet_Train, y_train )
else:
    model_final = load( "model/ada_clf.joblib" )
    
print ( model_final )

# Evaluate on test data

y_test_proba = model_final.predict_proba( DataSet_Test )[:,1]
print ( 'predict proba--> ', y_test_proba )

prec, rec, thresh = precision_recall_curve( y_test, y_test_proba)
bidx = np.argmax(prec * rec)
prob_cut_ = thresh[bidx]

print ( "Prob. cut: {}".format( prob_cut_ ) )

y_test_pred = ( y_test_proba >= prob_cut_ ).astype( "int32" )
print ( 'y_test_pred--> ', y_test_pred )

from sklearn.metrics import accuracy_score
print ( accuracy_score( y_test, y_test_pred ) )
print ( accuracy_score( y_test[ y_test == 1 ], y_test_pred[ y_test == 1 ] ) )
print ( accuracy_score( y_test[ y_test == 0 ], y_test_pred[ y_test == 0 ] ) )

test_errors = []
for test_predict_proba in model_final.staged_predict_proba( DataSet_Test ):
    test_errors.append( 1. - accuracy_score( ( test_predict_proba[:,1] >= prob_cut_ ), y_test ) )

n_trees = len( model_final )

estimator_errors = model_final.estimator_errors_[:n_trees]

print ( test_errors )
print ( estimator_errors )

# Save model

if train_model and save_model:
    import time
    id_ = time.strftime("%Y_%m_%d-%H_%M_%S")
    fileName_ = "adaboost_clf_{}_{}.joblib".format( label_anomalo, id_ )
    print ( "Saving model to {}".format( fileName_ ) )
    dump( model_final, fileName_ )
