#!/bin/sh

echo 'Running...'
echo python /afs/cern.ch/user/m/matheus/effici_WWCEP/effi_wjets.py
python /afs/cern.ch/user/m/matheus/effici_WWCEP/effi_wjets.py

cp *.h5 /eos/home-m/matheus/SWAN_projects/Acoplamento_Quartico_Anomalo

