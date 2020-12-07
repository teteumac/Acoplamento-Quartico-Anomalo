from __future__ import division, print_function
import h5py
import argparse
from almir import *

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
    
columns = np.array(  ['Mww','Pt_W_lep','dPhi_Whad_Wlep','dPhi_jatos_MET','jetAK8_pt','jetAK8_eta',
                      'jetAK8_prunedMass','jetAK8_tau21','METPt','muon_pt','muon_eta','ExtraTracks'], dtype = 'S')
    
#print(DataFrame)    
with h5py.File( 'output-' + label_ + '.h5', 'w') as f:
   dset = f.create_dataset( 'dados', data = DataFrame )
   dset_columns = f.create_dataset( 'columns', data = columns )
    
