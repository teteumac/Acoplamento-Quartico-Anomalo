from __future__ import division, print_function
import uproot4
import uproot
import awkward1 as ak
import numpy as np
import pandas as pd
import numba as nb
import scipy.constants
import uproot_methods.convert
import matplotlib.pyplot as plt
import mplhep as hep
from math import *
import sys
import uproot_methods
#from ROOT import *

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

norm_SM = ( cross_section_SM * Luminosity ) /  events_number_SM 
norm_ANOMALO1 = ( cross_section_ANOMALO1 * Luminosity ) /  events_number_ANOMALO1 
norm_ANOMALO2 = ( cross_section_ANOMALO2 * Luminosity ) /  events_number_ANOMALO2 
norm_ANOMALO3 = ( cross_section_ANOMALO3 * Luminosity ) /  events_number_ANOMALO3 
norm_ANOMALO4 = ( cross_section_ANOMALO4 * Luminosity ) /  events_number_ANOMALO4 
norm_ANOMALO5 = ( cross_section_ANOMALO5 * Luminosity ) /  events_number_ANOMALO5 
norm_ANOMALO6 = ( cross_section_ANOMALO6 * Luminosity ) /  events_number_ANOMALO6 
norm_ANOMALO7 = ( cross_section_ANOMALO7 * Luminosity ) /  events_number_ANOMALO7 
norm_ANOMALO8 = ( cross_section_ANOMALO8 * Luminosity ) /  events_number_ANOMALO8 

def open_files( file ): # Read and open the nTuple's TTree
    #print( file )
    root_ = uproot.open( file ) # Open the file 
    tree_ = root_[ "demo/Events" ] # Path to nTuple's TTree
    #print(tree_.show()) # Print all nTuple's branches on the screen
    return tree_

tree_SM       = open_files( PATH + SM )
tree_ANOMALO1 = open_files( PATH + ANOMALO1 )
tree_ANOMALO2 = open_files( PATH + ANOMALO2 )
tree_ANOMALO3 = open_files( PATH + ANOMALO3 )
tree_ANOMALO4 = open_files( PATH + ANOMALO4 )
tree_ANOMALO5 = open_files( PATH + ANOMALO5 )
tree_ANOMALO6 = open_files( PATH + ANOMALO6 )
tree_ANOMALO7 = open_files( PATH + ANOMALO7 )
tree_ANOMALO8 = open_files( PATH + ANOMALO8 )

def get_branch( tree , array ): # Return the disered branch 
    branch = pd.DataFrame( tree.array( array ) )[0] 
    return branch

 
# Put get_branch inside almir


def almir( tree ): # Return a DataFrame that contains the information about WW invariant mass, lepton pair p_T and DeltaPli between jet_MET and leptonic-W_hadronic-W
    Mw = 80.379 # Boson W mass
    k = ( ( Mw**2 ) / 2 + get_branch(tree,'muon_px')*get_branch(tree,'METPx')) + (get_branch(tree,'muon_py')*get_branch(tree,'METPy') ) 
    raiz = ( ( ( (k * get_branch(tree,'muon_pz'))**2) / (get_branch(tree,'muon_pt')**4)  - ( (get_branch(tree,'muon_E')*get_branch(tree,'METPt'))**2 - k)/get_branch(tree,'muon_pt')**2)**0.5 ).fillna(0) # .fillna(0) - replaces DataFrame NaN's that presents imaginary roots by 0 
    Pz_nu = ( ( k*get_branch(tree,'muon_pz') / (get_branch(tree,'muon_pt')**2 ) ) + raiz ) # Reconstructed neutrino's momentum z-component 
    W_lep_energy = get_branch(tree,'muon_E') + (get_branch(tree,'METPx')**2 + get_branch(tree,'METPy')**2 + Pz_nu**2)**0.5 # Lepton pair energy   
    TLV_lep = uproot_methods.TLorentzVectorArray(get_branch(tree,'muon_px')+get_branch(tree,'METPx'),get_branch(tree,'muon_py')+get_branch(tree,'METPy'),get_branch(tree,'muon_pz')+ Pz_nu,W_lep_energy) # Lepton pair 4-vector
    TLV_jet = uproot_methods.TLorentzVectorArray(get_branch(tree,'jetAK8_px'),get_branch(tree,'jetAK8_py'),get_branch(tree,'jetAK8_pz'),get_branch(tree,'jetAK8_E'))
    
    W_mass = ( TLV_lep + TLV_jet ).mass # WW invariant mass
    W_lep_pt = ( TLV_lep ).pt # Lepton pair p_T
    
    dphi_jet_lep = TLV_lep.phi - TLV_jet.phi
    dphi_jet_lep = np.where( dphi_jet_lep >=  scipy.constants.pi, dphi_jet_lep - 2*scipy.constants.pi, dphi_jet_lep)
    dphi_jet_lep = np.where( dphi_jet_lep <  -scipy.constants.pi, dphi_jet_lep + 2*scipy.constants.pi, dphi_jet_lep) # delta phi between the jet and the lepton pair
    dphi_jet_MET = get_branch(tree,'METphi') - TLV_jet.phi
    dphi_jet_MET = np.where( dphi_jet_MET >=  scipy.constants.pi, dphi_jet_MET - 2*scipy.constants.pi, dphi_jet_MET)
    dphi_jet_MET = np.where( dphi_jet_MET <  -scipy.constants.pi, dphi_jet_MET + 2*scipy.constants.pi, dphi_jet_MET) # delta phi between the jet e the MET 
    
    jetAK8_pt = TLV_jet.pt
    jetAK8_prunedMass = get_branche( tree , 'jetAK8_prunedMass')
    jetAK8_tau21 = np.array(get_branche(tree, 'jetAK8_tau21'))
    jetAK8_eta = TLV_jet.eta

    METPt = get_branche(tree, 'METPt')
    muon_pt = get_branche(tree, 'muon_pt')
    muon_eta = get_branche(tree, 'muon_eta')
    muon_phi = get_branche(tree, 'muon_phi')

    '''
    ** numbering the columns of the numpy array ** (to make cutting easier)

    0  --> WW mass
    1  --> Leptonic W pt
    2  --> DeltaPhi between W_hadronic e W_leptonic
    3  --> DeltaPhi between Jets e o MET
    4  --> jetAK8_pt
    5  --> jetAK8_eta
    6  --> jetAK8_prunedMass
    7  --> jetAK8_tau21
    8  --> METPt
    9  --> muon_pt
    10 --> muon_eta
    '''

    columns = ['Mww','Pt_W_lep','dPhi_Whad_Wlep','dPhi_jatos_MET','jetAK8_pt','jetAK8_eta','jetAK8_prunedMass','jetAK8_tau21','METPt','muon_pt','muon_eta']

    events_all = np.concatenate( ( W_mass.reshape(-1,1), W_lep_pt.reshape(-1,1), dphi_jet_lep.reshape(-1,1), 
    dphi_jet_MET.reshape(-1,1),jetAK8_pt.reshape(-1,1), jetAK8_eta.reshape(-1,1), jetAK8_prunedMass.reshape(-1,1), jetAK8_tau21.reshape(-1,1), 
    METPt.reshape(-1,1), muon_pt.reshape(-1,1), muon_eta.reshape(-1,1) ) , axis = 1 ) # concatenating all variables
    
    events_all_cut = (events_all[:,4] >= 200) & (events_all[:,5] <= 2.4) & (events_all[:,8] >= 40) & (events_all[:,9] >= 53) & (events_all[:,10] <= 2.4)  # making the cuts in the variables
    
    DataFrame = pd.DataFrame( events_all[ events_all_cut ] , columns = columns )

    return DataFrame


def plot(list_1,list_2,bins0,bins1,label0,label1,fontsize_leg,xmin,xmax,xlabel,ylabel,fontsize_xlabel,fontsize_ylabel,loc_leg,list_norm1,list_norm2):
    fig, axes = plt.subplots( 1, 2, figsize=(10,10) )
    axes[0].hist( list_1, bins = bins0, stacked=False, histtype = 'step', label=label0, density = False, weights = list_norm1, color = ['cyan', 'green', 'red', 'fuchsia','gold'] )
    axes[0].legend(loc=loc_leg, fontsize=fontsize_leg)
    axes[0].set_xlim(xmin,xmax)
    axes[0].set_ylabel(ylabel, fontsize = fontsize_ylabel)
    axes[0].set_yscale('log')
    axes[0] = hep.cms.label(data=False, paper=False, year='$9.792 fb^{-1}$', ax = axes[0])

    axes[1].hist( list_2, bins = bins1, stacked=False, histtype = 'step', label=label1, density = False, weights = list_norm2, color = ['cyan', 'green', 'red', 'fuchsia','gold'] )
    axes[1].legend(loc=loc_leg, fontsize=fontsize_leg)
    axes[1].set_xlim(xmin,xmax)
    axes[1].set_xlabel(xlabel,fontsize = fontsize_xlabel)
    axes[1].set_yscale('log')
    axes[1] = hep.cms.label(data=False, paper=False, year='$9.792 fb^{-1}$', ax = axes[1])    
    #plt.savefig(PATH_PLOT+'/{}.pdf'.format(name))
    plt.show()

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
'''
label = [
r'WWCEP $\alpha_{0}^{W}/\Lambda^{2} = 0.0 ;\alpha_{C}^{W}/\Lambda^{2}=0.0$ (SM)',
r'WWCEP $\alpha_{0}^{W}/\Lambda^{2} = 0.0 ;\alpha_{C}^{W}/\Lambda^{2}=2 \times 10^{-5}$',
r'WWCEP $\alpha_{0}^{W}/\Lambda^{2} = 0.0 ;\alpha_{C}^{W}/\Lambda^{2}=2 \times 10^{-6}$',
r'WWCEP $\alpha_{0}^{W}/\Lambda^{2} = 0.0 ;\alpha_{C}^{W}/\Lambda^{2}=5 \times 10^{-6}$', 
r'WWCEP $\alpha_{0}^{W}/\Lambda^{2} = 0.0 ;\alpha_{C}^{W}/\Lambda^{2}=8 \times 10^{-6}$',
r'WWCEP $\alpha_{0}^{W}/\Lambda^{2} = 0.5 \times 10^{-6} ;\alpha_{C}^{W}/\Lambda^{2}=0.0$',
r'WWCEP $\alpha_{0}^{W}/\Lambda^{2} = 1.0 \times 10^{-6} ;\alpha_{C}^{W}/\Lambda^{2}=0.0$',
r'WWCEP $\alpha_{0}^{W}/\Lambda^{2} = 2.0 \times 10^{-6} ;\alpha_{C}^{W}/\Lambda^{2}=0.0$',
r'WWCEP $\alpha_{0}^{W}/\Lambda^{2} = 5.0 \times 10^{-6} ;\alpha_{C}^{W}/\Lambda^{2}=0.0$'
        ]             
'''

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

# ---------------- Muon_{p_T} plot ---------------- #

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

sys.exit() # Stop the code

jetAK8_phi_0 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_0.0.root', 'jetAK8_phi')
jetAK8_phi_1 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_2e-5.root', 'jetAK8_phi')
jetAK8_phi_2 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_2e-6.root','jetAK8_phi')
jetAK8_phi_3 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_5e-6.root','jetAK8_phi')
jetAK8_phi_4 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_8e-6.root','jetAK8_phi')
jetAK8_phi_5 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.5e-6_0.0.root','jetAK8_phi')
jetAK8_phi_6 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_1.0e-6_0.0.root','jetAK8_phi')
jetAK8_phi_7 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_2.0e-6_0.0.root','jetAK8_phi')
jetAK8_phi_8 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_5.0e-6_0.0.root','jetAK8_phi')

plt.hist( [ jetAK8_phi_0,jetAK8_phi_1,jetAK8_phi_2,jetAK8_phi_3,jetAK8_phi_4,jetAK8_phi_5,jetAK8_phi_6,jetAK8_phi_7,jetAK8_phi_8 ], bins = 35, histtype = 'step', label=label, density = True)
plt.legend(loc='lower center', fontsize=12)
plt.xlabel(r'AK8$jet_{\phi}$')
plt.ylabel('Probability Density')
plt.style.use(hep.style.CMS)
hep.cms.label(data=False, paper=False, year='$9.792 fb^{-1}$')
plt.tight_layout()
plt.show()

jetAK8_eta_0 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_0.0.root', 'jetAK8_eta')
jetAK8_eta_1 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_2e-5.root', 'jetAK8_eta')
jetAK8_eta_2 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_2e-6.root','jetAK8_eta')
jetAK8_eta_3 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_5e-6.root','jetAK8_eta')
jetAK8_eta_4 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_8e-6.root','jetAK8_eta')
jetAK8_eta_5 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.5e-6_0.0.root','jetAK8_eta')
jetAK8_eta_6 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_1.0e-6_0.0.root','jetAK8_eta')
jetAK8_eta_7 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_2.0e-6_0.0.root','jetAK8_eta')
jetAK8_eta_8 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_5.0e-6_0.0.root','jetAK8_eta')

plt.hist( [ jetAK8_eta_0,jetAK8_eta_1,jetAK8_eta_2,jetAK8_eta_3,jetAK8_eta_4,jetAK8_eta_5,jetAK8_eta_6,jetAK8_eta_7,jetAK8_eta_8 ], bins = 35, histtype = 'step', label=label, density = True)
plt.legend(loc='best', fontsize=11)
plt.xlabel(r'AK8$jet_{\eta}$')
plt.ylabel('Probability Density')
plt.xlim(-3.0,3.0)
plt.style.use(hep.style.CMS)
hep.cms.label(data=False, paper=False, year='$9.792 fb^{-1}$')
plt.tight_layout()
plt.show()


METphi_0 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_0.0.root', 'METphi')
METphi_1 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_2e-5.root', 'METphi')
METphi_2 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_2e-6.root','METphi')
METphi_3 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_5e-6.root','METphi')
METphi_4 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_8e-6.root','METphi')
METphi_5 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.5e-6_0.0.root','METphi')
METphi_6 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_1.0e-6_0.0.root','METphi')
METphi_7 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_2.0e-6_0.0.root','METphi')
METphi_8 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_5.0e-6_0.0.root','METphi')

plt.hist( [ METphi_0,METphi_1,METphi_2,METphi_3,METphi_4,METphi_5,METphi_6,METphi_7,METphi_8 ], bins = 40, histtype = 'step', label=label, density = True)
plt.legend(loc='lower center', fontsize=11)
plt.xlabel(r'$MET_{\phi}$')
plt.ylabel('Probability Density')
#plt.xlim(0,1400)
plt.style.use(hep.style.CMS)
hep.cms.label(data=False, paper=True, year='$9.792 fb^{-1}$')
plt.tight_layout()
plt.show()

muon_phi_0 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_0.0.root', 'muon_phi')
muon_phi_1 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_2e-5.root', 'muon_phi')
muon_phi_2 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_2e-6.root','muon_phi')
muon_phi_3 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_5e-6.root','muon_phi')
muon_phi_4 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_8e-6.root','muon_phi')
muon_phi_5 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.5e-6_0.0.root','muon_phi')
muon_phi_6 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_1.0e-6_0.0.root','muon_phi')
muon_phi_7 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_2.0e-6_0.0.root','muon_phi')
muon_phi_8 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_5.0e-6_0.0.root','muon_phi')

plt.hist( [ muon_phi_0,muon_phi_1,muon_phi_2,muon_phi_3,muon_phi_4,muon_phi_5,muon_phi_6,muon_phi_7,muon_phi_8 ], bins = 30, histtype = 'step', label=label, density = True)
plt.legend(loc='lower center', fontsize=12)
plt.xlabel(r'$\phi_{\mu}$')
plt.ylabel('Probability Density')
#plt.xlim(0,1000)
plt.style.use(hep.style.CMS)
hep.cms.label(data=False, paper=False, year='$9.792 fb^{-1}$')
plt.tight_layout()
plt.show()

muon_eta_0 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_0.0.root', 'muon_eta')
muon_eta_1 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_2e-5.root', 'muon_eta')
muon_eta_2 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_2e-6.root','muon_eta')
muon_eta_3 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_5e-6.root','muon_eta')
muon_eta_4 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.0_8e-6.root','muon_eta')
muon_eta_5 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_0.5e-6_0.0.root','muon_eta')
muon_eta_6 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_1.0e-6_0.0.root','muon_eta')
muon_eta_7 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_2.0e-6_0.0.root','muon_eta')
muon_eta_8 = open_files(PATH + 'pre_MiniAOD_FPMC_WW_13TeV_5.0e-6_0.0.root','muon_eta')

plt.hist( [ muon_eta_0,muon_eta_1,muon_eta_2,muon_eta_3,muon_eta_4,muon_eta_5,muon_eta_6,muon_eta_7,muon_eta_8 ], bins = 30, histtype = 'step', label=label, density = True)
plt.legend(loc='best', fontsize=12)
#plt.plot([2.4,2.4],[0,0.6])
plt.xlabel(r'$\eta_{\mu}$')
plt.ylabel('Probability Density')
#plt.xlim(0,1000)
plt.style.use(hep.style.CMS)
hep.cms.label(data=False, paper=False, year='$9.792 fb^{-1}$')
plt.tight_layout()
plt.show()

