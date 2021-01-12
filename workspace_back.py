
from __future__ import division, print_function
import h5py
import argparse
from macro_background import *
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

dataset_SistemaCentral = almir( fileNames_[0]  )

PATH = "/eos/home-m/matheus/amostras_2016/merged_06_10_20/" # Caminho comum para os arquivos .root

data_B = PATH + 'SingleMuonB.root'
data_C = PATH + 'SingleMuonC.root'
data_G = PATH + 'SingleMuonG.root'

# Concatenando todas as eras dos dados
era_B = soma_data( data_B )
era_C = soma_data( data_C )
era_G = soma_data( data_G )

dados_protons_erasBCG = np.concatenate( ( era_B , era_C , era_B ) , axis = 0 )

# Eliminando todas as linhas que tem NaN  
mask = np.any( np.isnan( dados_protons_erasBCG ) , axis = 1 )
dataset_PPS = dados_protons_erasBCG[ ~mask ]
dataset_PPS.shape


columns = ['Mww','Pt_W_lep','dPhi_Whad_Wlep','dPhi_jatos_MET','jetAK8_pt','jetAK8_eta',
    'jetAK8_prunedMass','jetAK8_tau21','METPt','muon_pt','muon_eta','ExtraTracks', 'PUWeight',
    'xi1','xi2','anguloX1', 'anguloX2', 'anguloY1','anguloY2', 'rpid1','rpid2','arm1','arm2','ismultirp1','ismultirp2']

dataset_PPS_SistemaCentral = r_sample(dataset_SistemaCentral,dataset_PPS) 
    
with h5py.File( 'output-' + label_ + '.h5', 'w') as f:
   dset = f.create_dataset( 'dados', data = dataset_PPS_SistemaCentral )
   dset_columns = f.create_dataset( 'columns', data = columns )


