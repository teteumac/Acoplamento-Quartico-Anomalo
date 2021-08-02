#!/bin/tcsh

source /afs/cern.ch/user/m/matheus/myenv/bin/activate.csh

set DataSet_Signal_=$1
set DataSet_Backgr_=$2
set DataSet_SM_=$3
set DataSet_Dados_=$4
set label_anomalo_=$5

echo "DataSet_Signal_: "$DataSet_Signal_
echo "DataSet_Backgr_: "$DataSet_Backgr_
echo "DataSet_SM_: "$DataSet_SM_
echo "DataSet_Dados_:"$DataSet_Dados_
echo "label_anomalo_:"$label_anomalo_

set EXEC = /afs/cern.ch/user/m/matheus/output_ML_binary
set OUTPUT = /afs/cern.ch/user/m/matheus/output_ML_binary/output_Bayesian_Optimization

echo $EXEC
echo $OUTPUT

if ( ! $?PYTHONPATH ) then
    setenv PYTHONPATH ${EXEC}
else
    setenv PYTHONPATH ${PYTHONPATH}:${EXEC}
endif
echo PYTHONPATH set to $PYTHONPATH
env

echo $PWD

echo 'Running...'
echo python3 $EXEC/Bayesian_Optimization.py --DataSet_Signal=$DataSet_Signal_ --DataSet_Backgr=$DataSet_Backgr_ --DataSet_SM=$DataSet_SM_ --DataSet_Dados=$DataSet_Dados_ --Label_Signal=$label_anomalo_

python3 $EXEC/Bayesian_Optimization.py --DataSet_Signal=$DataSet_Signal_ --DataSet_Backgr=$DataSet_Backgr_ --DataSet_SM=$DataSet_SM_ --DataSet_Dados=$DataSet_Dados_ --Label_Signal=$label_anomalo_

cp *.joblib $OUTPUT
