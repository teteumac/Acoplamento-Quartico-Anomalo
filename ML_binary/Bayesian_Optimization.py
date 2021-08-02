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

import argparse

from joblib import dump

PATH = '/eos/home-m/matheus/SWAN_projects/Acoplamento_Quartico_Anomalo/'

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

plt.style.use(hep.style.ROOT)

def open_file_MC( file ):
    df = None
    with h5py.File( file , 'r' ) as f:
        dset = f[ 'dados' ]
        array = np.array( dset )
        dataframe = pd.DataFrame( array , columns = ['Mww', 'Pt_W_lep', 'dPhi_Whad_Wlep', 'dPhi_jatos_MET', 'jetAK8_pt','jetAK8_eta', 'jetAK8_prunedMass','jetAK8_tau21', 'METPt', 'muon_pt', 'muon_eta', 'ExtraTracks', 'PUWeight', 'Yww', 'xi1', 'xi2', 'Mx', 'Yx','Mww/Mx', 'Yww_Yx', 'weight'] )
        return dataframe

def open_file_datadriven( file ):
    df = None
    with h5py.File( file , 'r' ) as f:
        dset = f[ 'dados' ]
        array = np.array( dset )
        dataframe = pd.DataFrame( array , columns = ['Mww','Pt_W_lep','dPhi_Whad_Wlep','dPhi_jatos_MET','jetAK8_pt','jetAK8_eta',
       'jetAK8_prunedMass','jetAK8_tau21','METPt','muon_pt','muon_eta','ExtraTracks', 'Yww',
       'xi1','xi2', 'Mx','Yx', 'Mww/Mx','Yww_Yx', 'weight' ]  )
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

select_columns = ['Mww', 'Pt_W_lep', 'jetAK8_pt','jetAK8_eta', 'jetAK8_tau21', 'METPt', 'muon_pt', 'muon_eta', 'ExtraTracks', 'Yww', 'xi1', 'xi2','Mx', 'Mww/Mx']

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

data_set_dados_multirp = open_file_Data( PATH + DataSet_Dados_ )

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

print('DataSet Train',DataSet_Train_)

DataSet_Train = DataSet_Train_[select_columns] 
DataSet_Test  = DataSet_Test_[select_columns] 

from skopt import gp_minimize

def procurar_param(params):
    learning_rate = params[0]
    num_leaves = params[1]
    min_child_samples = params[2]
    subsample = params[3]
    colsample_bytree = params[4]
    n_estimators = params[5]
    max_depth = params[6]

    print(params, '\n')

    mdl = LGBMClassifier( learning_rate = learning_rate, num_leaves = num_leaves, min_child_samples = min_child_samples, subsample = subsample, colsample_bytree = colsample_bytree, random_state = 42, subsample_freq = 1, n_estimators = n_estimators, max_depth = max_depth, )
    
    mdl.fit(DataSet_Train, y_train)

    p = mdl.predict_proba(DataSet_Test)[:,1]

    return -roc_auc_score(y_test, p)


space = [(1e-3, 1e-1, 'log-uniform'), #learning rate
         (2, 128), # num_leaves
         (2, 100), # min_child_samples
         (0.05, 1.0), # subsample
         (0.1, 1.0),# colsample bytree
         (100,1000), # n_estimators
         (2,100)] # max_depht

resultados_gp = gp_minimize(procurar_param, space, random_state=42, verbose=1, n_calls=200, n_random_starts=100)
import time

def treinar_modelo( params, verbose = 1 ):
    learning_rate = params[0]
    num_leaves = params[1]
    min_child_samples = params[2]
    subsample = params[3]
    colsample_bytree = params[4]
    n_estimators = params[5]
    max_depth = [6]
    
    print(params, '\n')
    
    mdl = LGBMClassifier( learning_rate=learning_rate, num_leaves=num_leaves, min_child_samples=min_child_samples, subsample=subsample, colsample_bytree=colsample_bytree, random_state=42, subsample_freq=1,  n_estimators=n_estimators, max_depth=max_depth )
    mdl.fit(DataSet_Train, y_train)
        
    id_ = time.strftime("%Y_%m_%d-%H_%M_%S")
    fileName_ = "LGBM_clf_Bayesian_{}_{}.joblib".format( label_anomalo, id_ )	
    print ( "Saving model to {}".format( fileName_ ) )
    dump( mdl, fileName_ )

    y_probs = mdl.predict_proba(DataSet_Test)[:,1]
    prec, rec, thresh = precision_recall_curve(y_test, y_probs)
    bidx = np.argmax(prec * rec)
    best_cut = thresh[bidx]
    print( 'best_cut', best_cut )

    preds = y_probs >= best_cut

    if verbose == 1:

        fpr, tpr, thresholds = roc_curve(y_test, y_probs, drop_intermediate=False)
        print(f'AUC = {auc(fpr, tpr)}')
        print("Purity in test sample     : {:2.2f}%".format(100 * precision_score(y_test, preds)))
        print("Efficiency in test sample : {:2.2f}%".format(100 * recall_score(y_test, preds)))
        print("Accuracy in test sample   : {:2.2f}%".format(100 * accuracy_score(y_test, preds)))

    return y_probs,best_cut

SaveFig = '/afs/cern.ch/user/m/matheus/output_ML_binary/output_Bayesian_Optimization/'
from sklearn.metrics import precision_recall_curve,roc_curve,auc,precision_score,recall_score,accuracy_score


predict_proba, best_cut = treinar_modelo( resultados_gp.x )

print('Best Params', resultados_gp.x )
print('\n')
print('Best Cut {}'.format(label_anomalo), best_cut)
