from __future__ import division, print_function


import h5py
import argparse
from macro_dados import *
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


with h5py.File( 'output-' + label_ + '.h5', 'w') as f:
    dset = f.create_dataset( 'dados', data = DataFrame )

print( 'Finish...'  )
