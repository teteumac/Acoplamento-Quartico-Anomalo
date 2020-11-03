from __future__ import division, print_function
from almir import *


# signal files separated by (_) --> alpha0 _ alphaC 

PATH      = '/mnt/hadoop/cms/store/user/mthiel/samples/samples_2016/merged_06_10_20/' # input nTuples path 
PATH_PLOT = '/home/malvesga/WW/Acoplamento-Quartico-Anomalo/Miguel/plots_signal/' # output plots path

SM        = 'pre_MiniAOD_FPMC_WW_13TeV_0.0_0.0.root'
ANOMALO1  = 'pre_MiniAOD_FPMC_WW_13TeV_0.0_2e-5.root'
ANOMALO2  = 'pre_MiniAOD_FPMC_WW_13TeV_0.0_2e-6.root'
ANOMALO3  = 'pre_MiniAOD_FPMC_WW_13TeV_0.0_5e-6.root'
ANOMALO4  = 'pre_MiniAOD_FPMC_WW_13TeV_0.0_8e-6.root'
ANOMALO5  = 'pre_MiniAOD_FPMC_WW_13TeV_0.5e-6_0.0.root'
ANOMALO6  = 'pre_MiniAOD_FPMC_WW_13TeV_1.0e-6_0.0.root'
ANOMALO7  = 'pre_MiniAOD_FPMC_WW_13TeV_2.0e-6_0.0.root'
ANOMALO8  = 'pre_MiniAOD_FPMC_WW_13TeV_5.0e-6_0.0.root'


# -------------------------------------------------------------------- #

# SIGNAL Monte Carlo samples - Cross Sections #

# -------------------------------------------------------------------- #

cross_section_SM       = 40.41*0.17
cross_section_ANOMALO1 = 166.1*0.17 
cross_section_ANOMALO2 = 41.90*0.17
cross_section_ANOMALO3 = 48.75*0.17
cross_section_ANOMALO4 = 61.14*0.17
cross_section_ANOMALO5 = 41.58*0.17
cross_section_ANOMALO6 = 44.93*0.17
cross_section_ANOMALO7 = 58.18*0.17 
cross_section_ANOMALO8 = 150.3*0.17

# -------------------------------------------------------------------- #

# SIGNAL Monte Carlo samples - Events Number #

# -------------------------------------------------------------------- #

events_number_SM       = 35000
events_number_ANOMALO1 = 35000
events_number_ANOMALO2 = 35000
events_number_ANOMALO3 = 35000
events_number_ANOMALO4 = 35000
events_number_ANOMALO5 = 35000
events_number_ANOMALO6 = 35000
events_number_ANOMALO7 = 35000
events_number_ANOMALO8 = 35000


# -------------------------------------------------------------------- #

# Data events - Luminosity #

# -------------------------------------------------------------------- #

SingleMuon_Run2016B = 4.55
SingleMuon_Run2016C = 1.59
SingleMuon_Run2016G = 3.65
Luminosity        = SingleMuon_Run2016B + SingleMuon_Run2016C + SingleMuon_Run2016G 


# SIGNAL events - Normalization #
norm_SM = ( cross_section_SM * Luminosidade ) / ( number_events_SM )
norm_ANOMALO1 = ( cross_section_ANOMALO1 * Luminosidade ) / ( number_events_ANOMALO1 )
norm_ANOMALO2 = ( cross_section_ANOMALO2 * Luminosidade ) / ( number_events_ANOMALO2 )
norm_ANOMALO3 = ( cross_section_ANOMALO3 * Luminosidade ) / ( number_events_ANOMALO3 )
norm_ANOMALO4 = ( cross_section_ANOMALO4 * Luminosidade ) / ( number_events_ANOMALO4 )
norm_ANOMALO5 = ( cross_section_ANOMALO5 * Luminosidade ) / ( number_events_ANOMALO5 )
norm_ANOMALO6 = ( cross_section_ANOMALO6 * Luminosidade ) / ( number_events_ANOMALO6 )
norm_ANOMALO7 = ( cross_section_ANOMALO7 * Luminosidade ) / ( number_events_ANOMALO7 )
norm_ANOMALO8 = ( cross_section_ANOMALO8 * Luminosidade ) / ( number_events_ANOMALO8 )

tree_SM       = open_files( PATH + SM )
tree_ANOMALO1 = open_files( PATH + ANOMALO1 )
tree_ANOMALO2 = open_files( PATH + ANOMALO2 )
tree_ANOMALO3 = open_files( PATH + ANOMALO3 )
tree_ANOMALO4 = open_files( PATH + ANOMALO4 )
tree_ANOMALO5 = open_files( PATH + ANOMALO5 )
tree_ANOMALO6 = open_files( PATH + ANOMALO6 )
tree_ANOMALO7 = open_files( PATH + ANOMALO7 )
tree_ANOMALO8 = open_files( PATH + ANOMALO8 )

label_0 = [
r'WWCEP $\alpha_{C}^{W}/\Lambda^{2}=0.0$ (SM)',
r'WWCEP $\alpha_{C}^{W}/\Lambda^{2}=2 \times 10^{-5}$',
r'WWCEP $\alpha_{C}^{W}/\Lambda^{2}=2 \times 10^{-6}$',
r'WWCEP $\alpha_{C}^{W}/\Lambda^{2}=5 \times 10^{-6}$', 
r'WWCEP $\alpha_{C}^{W}/\Lambda^{2}=8 \times 10^{-6}$'
          ]

label_1 = [
r'WWCEP $\alpha_{0}^{W}/\Lambda^{2} = 0.0$ (SM)',  
r'WWCEP $\alpha_{0}^{W}/\Lambda^{2} = 0.5 \times 10^{-6} $',
r'WWCEP $\alpha_{0}^{W}/\Lambda^{2} = 1.0 \times 10^{-6} $',
r'WWCEP $\alpha_{0}^{W}/\Lambda^{2} = 2.0 \times 10^{-6} $',
r'WWCEP $\alpha_{0}^{W}/\Lambda^{2} = 5.0 \times 10^{-6} $'
          ]     

DataFrame_SM       = almir(tree_SM)
DataFrame_ANOMALO1 = almir(tree_ANOMALO1)
DataFrame_ANOMALO2 = almir(tree_ANOMALO2)
DataFrame_ANOMALO3 = almir(tree_ANOMALO3)
DataFrame_ANOMALO4 = almir(tree_ANOMALO4)
DataFrame_ANOMALO5 = almir(tree_ANOMALO5)
DataFrame_ANOMALO6 = almir(tree_ANOMALO6)
DataFrame_ANOMALO7 = almir(tree_ANOMALO7)
DataFrame_ANOMALO8 = almir(tree_ANOMALO8)

lista_norm_signal_1 = [[norm_SM]*len(DataFrame_SM),[norm_ANOMALO1]*len(DataFrame_ANOMALO1),[norm_ANOMALO2]*len(DataFrame_ANOMALO2),[norm_ANOMALO3]*len(DataFrame_ANOMALO3),[norm_ANOMALO4]*len(DataFrame_ANOMALO4)]
lista_norm_signal_2 = [[norm_SM]*len(DataFrame_SM),[norm_ANOMALO5]*len(DataFrame_ANOMALO5),[norm_ANOMALO6]*len(DataFrame_ANOMALO6),[norm_ANOMALO7]*len(DataFrame_ANOMALO7),[norm_ANOMALO8]*len(DataFrame_ANOMALO8)]

# ---------------- jetAK8_{p_T} plot ---------------- #

jetAK8_pt_0 = [
DataFrame_SM['jetAK8_pt'],
DataFrame_ANOMALO1['jetAK8_pt'],
DataFrame_ANOMALO2['jetAK8_pt'],
DataFrame_ANOMALO3['jetAK8_pt'],
DataFrame_ANOMALO4['jetAK8_pt'] ]
jetAK8_pt_1 = [
DataFrame_SM['jetAK8_pt'],
DataFrame_ANOMALO5['jetAK8_pt'],
DataFrame_ANOMALO6['jetAK8_pt'],
DataFrame_ANOMALO7['jetAK8_pt'],
DataFrame_ANOMALO8['jetAK8_pt'] ]

list_norm_signal_1 = [[norm_SM]*len(jetAK8_pt_0[0]),[norm_ANOMALO1]*len(jetAK8_pt_0[1]),[norm_ANOMALO2]*len(jetAK8_pt_0[2]),[norm_ANOMALO3]*len(jetAK8_pt_0[3]),[norm_ANOMALO4]*len(jetAK8_pt_0[4])]
list_norm_signal_2 = [[norm_SM]*len(jetAK8_pt_1[0]),[norm_ANOMALO5]*len(jetAK8_pt_1[1]),[norm_ANOMALO6]*len(jetAK8_pt_1[2]),[norm_ANOMALO7]*len(jetAK8_pt_1[3]),[norm_ANOMALO8]*len(jetAK8_pt_1[4])]


plot(jetAK8_pt_0,jetAK8_pt_1,140,140,label_0,label_1,9,100,1400,r'$p_{T_{AK8Jet}}$(GeV)','Events',19,19,'lower left',list_norm_signal_1,list_norm_signal_2)

# ---------------- Mass_{WW} plot ---------------- #

Mww_0 = [ 
DataFrame_SM['Mww'], 
DataFrame_ANOMALO1['Mww'], 
DataFrame_ANOMALO2['Mww'], 
DataFrame_ANOMALO3['Mww'], 
DataFrame_ANOMALO4['Mww']
]
Mww_1 = [ 
DataFrame_SM['Mww'], 
DataFrame_ANOMALO5['Mww'], 
DataFrame_ANOMALO6['Mww'], 
DataFrame_ANOMALO7['Mww'], 
DataFrame_ANOMALO8['Mww']
]


plot(Mww_0,Mww_1,140,140,label_0,label_1,8,0,3100,r'$M_{WW}$(GeV)','Events',19,19,'lower left',list_norm_signal_1,list_norm_signal_2)


# ---------------- Leptonic W plot ---------------- #

W_lep_Pt_0 = [ 
DataFrame_SM['Pt_W_lep'], 
DataFrame_ANOMALO1['Pt_W_lep'], 
DataFrame_ANOMALO2['Pt_W_lep'], 
DataFrame_ANOMALO3['Pt_W_lep'], 
DataFrame_ANOMALO4['Pt_W_lep']
]
W_lep_Pt_1 = [ 
DataFrame_SM['Pt_W_lep'], 
DataFrame_ANOMALO5['Pt_W_lep'], 
DataFrame_ANOMALO6['Pt_W_lep'], 
DataFrame_ANOMALO7['Pt_W_lep'], 
DataFrame_ANOMALO8['Pt_W_lep']
]

plot(W_lep_Pt_0,W_lep_Pt_1,2000,1000,label_0,label_1,8,0,1300,r'${p_{T_{W_{leptonic}}}}(GeV)$','Events',19,19,'lower left',list_norm_signal_1,list_norm_signal_2)

# ---------------- Delta Phi (jet,lep) plot ---------------- #

Dphi_jet_lep_0 = [ 
DataFrame_SM['dPhi_Whad_Wlep'], 
DataFrame_ANOMALO1['dPhi_Whad_Wlep'], 
DataFrame_ANOMALO2['dPhi_Whad_Wlep'], 
DataFrame_ANOMALO3['dPhi_Whad_Wlep'], 
DataFrame_ANOMALO4['dPhi_Whad_Wlep']
]
Dphi_jet_lep_1 = [ 
DataFrame_SM['dPhi_Whad_Wlep'], 
DataFrame_ANOMALO5['dPhi_Whad_Wlep'], 
DataFrame_ANOMALO6['dPhi_Whad_Wlep'], 
DataFrame_ANOMALO7['dPhi_Whad_Wlep'], 
DataFrame_ANOMALO8['dPhi_Whad_Wlep']
]

plot(Dphi_jet_lep_0,Dphi_jet_lep_1,35,35,label_0,label_1,10,-3.3,3.3,r'$\Delta \phi_{(hadro, lepto)}$','Events',19,19,'upper center',list_norm_signal_1,list_norm_signal_2)

# ---------------- Delta Phi (jet,MET) plot ---------------- #

Dphi_jet_MET_0 = [ 
DataFrame_SM['dPhi_jatos_MET'], 
DataFrame_ANOMALO1['dPhi_jatos_MET'], 
DataFrame_ANOMALO2['dPhi_jatos_MET'], 
DataFrame_ANOMALO3['dPhi_jatos_MET'], 
DataFrame_ANOMALO4['dPhi_jatos_MET']
]
Dphi_jet_MET_1 = [ 
DataFrame_SM['dPhi_jatos_MET'], 
DataFrame_ANOMALO5['dPhi_jatos_MET'], 
DataFrame_ANOMALO6['dPhi_jatos_MET'], 
DataFrame_ANOMALO7['dPhi_jatos_MET'], 
DataFrame_ANOMALO8['dPhi_jatos_MET']
]

plot(Dphi_jet_MET_0,Dphi_jet_MET_1,35,35,label_0,label_1,10,-3.3,3.3,r'$\Delta \phi_{(jet, MET)}$','Events',19,19,'upper center',list_norm_signal_1,list_norm_signal_2)

# ---------------- jetAK8_{prunedMass} plot ---------------- #

jetAK8_prunedMass_0 = [
DataFrame_SM['jetAK8_prunedMass'],
DataFrame_ANOMALO1['jetAK8_prunedMass'],
DataFrame_ANOMALO2['jetAK8_prunedMass'],
DataFrame_ANOMALO3['jetAK8_prunedMass'],
DataFrame_ANOMALO4['jetAK8_prunedMass'],
]
jetAK8_prunedMass_1 = [
DataFrame_SM['jetAK8_prunedMass'],
DataFrame_ANOMALO5['jetAK8_prunedMass'],
DataFrame_ANOMALO6['jetAK8_prunedMass'],
DataFrame_ANOMALO7['jetAK8_prunedMass'],
DataFrame_ANOMALO8['jetAK8_prunedMass'],
]
plot(jetAK8_prunedMass_0,jetAK8_prunedMass_1,80,80,label_0,label_1,10,0,180,r'$M_{pruned\;Jet}$(GeV)','Events',19,19,'best',list_norm_signal_1,list_norm_signal_2)

# ---------------- jetAK8_{tau_{21}} plot ---------------- #

jetAK8_tau21_0 = [
DataFrame_SM['jetAK8_tau21'],
DataFrame_ANOMALO1['jetAK8_tau21'],
DataFrame_ANOMALO2['jetAK8_tau21'],
DataFrame_ANOMALO3['jetAK8_tau21'],
DataFrame_ANOMALO4['jetAK8_tau21'],
]
jetAK8_tau21_1 = [
DataFrame_SM['jetAK8_tau21'],
DataFrame_ANOMALO5['jetAK8_tau21'],
DataFrame_ANOMALO6['jetAK8_tau21'],
DataFrame_ANOMALO7['jetAK8_tau21'],
DataFrame_ANOMALO8['jetAK8_tau21'],
]

plot(jetAK8_tau21_0,jetAK8_tau21_1,20,20,label_0,label_1,9,0,2.1,r'$\tau_{21}$','Events',19,19,'lower left',list_norm_signal_1,list_norm_signal_2)

# ---------------- MET_{p_T} plot ---------------- #

METPt_0 = [
DataFrame_SM['METPt'],
DataFrame_ANOMALO1['METPt'],
DataFrame_ANOMALO2['METPt'],
DataFrame_ANOMALO3['METPt'],
DataFrame_ANOMALO4['METPt'],
]
METPt_1 = [
DataFrame_SM['METPt'],
DataFrame_ANOMALO5['METPt'],
DataFrame_ANOMALO6['METPt'],
DataFrame_ANOMALO7['METPt'],
DataFrame_ANOMALO8['METPt'],
]

plot(METPt_0,METPt_1,180,180,label_0,label_1,9,0,1300,r'${p_{T_{MET}}}$(GeV)','Events',19,19,'lower left',list_norm_signal_1,list_norm_signal_2)

# ---------------- Extra Tracks plot ---------------- #

muon_pt_0 = [
DataFrame_SM['muon_pt'],
DataFrame_ANOMALO1['muon_pt'],
DataFrame_ANOMALO2['muon_pt'],
DataFrame_ANOMALO3['muon_pt'],
DataFrame_ANOMALO4['muon_pt'],
]
muon_pt_1 = [
DataFrame_SM['muon_pt'],
DataFrame_ANOMALO5['muon_pt'],
DataFrame_ANOMALO6['muon_pt'],
DataFrame_ANOMALO7['muon_pt'],
DataFrame_ANOMALO8['muon_pt'],
]

plot(muon_pt_0,muon_pt_1,1900,1800,label_0,label_1,9,0,1100,r'${p_{T_{\mu}}}(GeV)$','Events',19,19,'lower left',list_norm_signal_1,list_norm_signal_2)


