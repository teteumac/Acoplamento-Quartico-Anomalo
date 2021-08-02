from __future__ import division, print_function
import h5py
import argparse
from macro_background import *
import uproot
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description = 'Creates data table from ntuple')
parser.add_argument('--files', help = 'File paths' )
parser.add_argument('--label', help = 'Label suffix' )
args = parser.parse_args()

fileNames_ = args.files.split(",")
print( "Reading files: " )
for item in fileNames_: print ( 'item -->',item )
label_ = args.label
print ( "Label: " + label_ )

dataset_SistemaCentral = almir( fileNames_[0]  )

PATH = "/eos/home-m/matheus/amostras_2016/merged_06_10_20/" # Caminho comum para os arquivos .root

data_B = PATH + 'SingleMuonB.root'
data_C = PATH + 'SingleMuonC.root'
data_G = PATH + 'SingleMuonG.root'

# Concatenando todas as eras dos dados
era_B = soma_data( data_B )
era_C = soma_data( data_C )
era_G = soma_data( data_G )

dados_protons_erasBCG = np.concatenate( ( era_B , era_C , era_G ) , axis = 0 )


# Eliminando todas as linhas que tem NaN  
mask = np.any( np.isnan( dados_protons_erasBCG ) , axis = 1 )
print('mask',mask)
dataset_PPS = dados_protons_erasBCG[ ~mask ]
print('dataset_PPS::\n', dataset_PPS )

#MultiRP = ( dataset_PPS[:,10] == 1 ) & ( dataset_PPS[:,11] == 1 )  

#dataset_PPS = dataset_PPS[MultiRP]
#print( 'dataset_PPS so multiRP \n', pd.DataFrame(dataset_PPS) )

lista = [ ]
for _ in range(10):
    lista.append( r_sample( dataset_SistemaCentral, dados_protons_erasBCG ) )

dataset_PPS_SistemaCentral = np.array( np.concatenate( lista, axis = 0 ) )
#dataset_PPS_SistemaCentral = r_sample( dataset_SistemaCentral, dados_protons_erasBCG )
print( 'DataFrame PPS - Sistema Central \n', pd.DataFrame( dataset_PPS_SistemaCentral ), '\n' )    

print('shape dataset back\n', dataset_PPS_SistemaCentral.shape)

with h5py.File( 'output-' + label_ + '.h5', 'w') as f:
   dset = f.create_dataset( 'dados', data = dataset_PPS_SistemaCentral )

with h5py.File( 'output-Sistema_Central' + label_ + '.h5', 'w') as f:
   dset = f.create_dataset( 'dados', data = dataset_SistemaCentral )

print( 'Acabou de rodar')
