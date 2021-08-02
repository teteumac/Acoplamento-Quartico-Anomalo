#!/bin/sh

echo 'Running...'
echo python /afs/cern.ch/user/m/matheus/effici_WWCEP/effic_QCD.py
python /afs/cern.ch/user/m/matheus/effici_WWCEP/effic_QCD.py

cp *.h5 /eos/home-m/matheus/SWAN_projects/Acoplamento_Quartico_Anomalo

