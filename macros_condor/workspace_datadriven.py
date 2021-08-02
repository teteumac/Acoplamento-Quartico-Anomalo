
from __future__ import division, print_function


import h5py
import argparse
from macro_dados import *
import random
import uproot

parser = argparse.ArgumentParser(description = 'Creates data table from ntuple')
parser.add_argument('--files', help = 'File paths' )
parser.add_argument('--label', help = 'Label suffix' )
args = parser.parse_args()

fileNames_ = args.files.split(",")
print( "Reading files: " )
for item in fileNames_: print ( 'item -->',item )
label_ = args.label
print ( "Label: " + label_ )



DataFrame = almir( fileNames_[0]  )

print('DataFrame:: \n', pd.DataFrame( DataFrame) )

def r_sample(m_base, m_external):
    n_linhas_m_base = m_base.shape[0] # matriz com as variáveis do detector central
    n_linhas_m_external = m_external.shape[0] # matriz com as variáveis dos prótons
    index_m_external = [i for i in range(n_linhas_m_external)]
    mask = random.sample(index_m_external, k=n_linhas_m_base)
    m_sampled = np.concatenate((m_base, m_external[mask, :]), axis=1)
    return m_sampled

def soma_data(file):
    print(f'abrindo arquivo {file}')
    ProtCand_xi = open_files_protons(file,'ProtCand_xi')
    print('\t ProtCand_xi aberto')
    ProtCand_ismultirp = open_files_protons(file,'ProtCand_ismultirp')
    print('\t ProtCand_ismultirp aberto')
    m = np.concatenate( ( ProtCand_xi, ProtCand_ismultirp ) , axis = 1 )
    return m

PATH = "/eos/home-m/matheus/amostras_2016/merged_06_10_20/" # Caminho comum para os arquivos .root

data_B = PATH + 'SingleMuonB.root'
data_C = PATH + 'SingleMuonC.root'
data_G = PATH + 'SingleMuonG.root'

# Concatenando todas as eras dos dados
era_B = soma_data( data_B )
era_C = soma_data( data_C )
era_G = soma_data( data_G )

dados_protons_erasBCG = np.concatenate( ( era_B , era_C , era_G ) , axis = 0 )

mask = np.any( np.isnan( dados_protons_erasBCG ) , axis = 1 )
print('mask',mask)
dataset_PPS = dados_protons_erasBCG[ ~mask ]
print('dataset_PPS::', dataset_PPS )
              

# executa um número N de vezes a randomização dos prótons

lista = []
for _ in range(100):
    lista.append( r_sample( DataFrame, dados_protons_erasBCG ) )

DataSet = np.array( np.concatenate( lista, axis = 0) )

print('DataSet --> \n', DataSet.shape )
print('DataFrame --> \n', DataFrame.shape )


with h5py.File( 'output-' + label_ + '.h5', 'w') as f:
    dset = f.create_dataset( 'dados', data = DataSet )

print( 'Finish...'  )

