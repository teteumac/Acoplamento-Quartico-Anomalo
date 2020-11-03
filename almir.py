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
import h5py

def open_files( file ): # Funcao que ler e abre as trees das nTuplas
    #print( file )
    root_ = uproot.open( file ) # abertura dos arquivos 
    tree_ = root_[ "demo/Events" ] # trees das nTupla
    #print(tree_.show()) # printar na tela todos os branches da nTupla
    return tree_

def get_branche( tree , array ): # Return the disered branch 
    branch = np.array( pd.DataFrame( tree.array( array ) )[0] )
    return branch

def get_PfFrom( tree , array ):
    branche = tree.array( array ) 
    return branche


def almir( tree ): # Funcao que retorna um DataFrame que contem a massa invariante do WW, pt do par de le
    

    Mw = 80.379 # massa do boson W
    
    jetAK8_pt = get_branche( tree , 'jetAK8_pt')
    jetAK8_prunedMass = get_branche( tree , 'jetAK8_prunedMass')
    jetAK8_tau21 = get_branche(tree, 'jetAK8_tau21')
    jetAK8_eta = get_branche( tree , 'jetAK8_eta')
    jetAK8_px =  get_branche(tree,'jetAK8_px')
    jetAK8_py =  get_branche(tree,'jetAK8_py')
    jetAK8_pz =  get_branche(tree,'jetAK8_pz')
    jetAK8_E =  get_branche(tree,'jetAK8_E')

    #print(jetAK8_pt) 
    
    METPt  = get_branche(tree, 'METPt' )
    METPx  = get_branche(tree, 'METPx' )
    METPy  = get_branche(tree, 'METPy' )
    METphi = get_branche(tree, 'METphi')

    #print(METphi)

    muon_pt  = get_branche(tree, 'muon_pt')
    muon_eta = get_branche(tree, 'muon_eta')
    muon_phi = get_branche(tree, 'muon_phi')
    muon_px  = get_branche(tree,'muon_px')
    muon_py  = get_branche(tree,'muon_py')
    muon_pz  = get_branche(tree,'muon_pz')
    muon_E   = get_branche(tree,'muon_E') 
    
    
    k = ( ( Mw**2 ) / 2 + muon_px * METPx ) +  (muon_py * METPy ) 
    raiz_ = ( ( ( (k * muon_pz)**2) / (muon_pt**4)  - ( (muon_E * METPt)**2 - k) / muon_pt**2)**0.5 )    
    raiz = np.nan_to_num(raiz_) # Os valores de NaN, causados pela divisão por 0 ou pelo resultado de uma raiz imaginária, é substituida por NaN
    Pz_nu = ( ( k * muon_pz / (muon_pt**2 ) ) + raiz ) # coordenada z do momentum do neutrino reconstruido
    W_lep_energy = ( muon_E + (METPx**2 + METPy**2 + Pz_nu**2)**0.5) # Energia do par de léptons  


    # Usamos o TLorenctzVector do Python 
    TLV_lep = uproot_methods.TLorentzVectorArray(
              muon_px + METPx,
              muon_py + METPy,
              muon_pz + Pz_nu, 
              W_lep_energy) # 4-vector do par de lepton
            
    TLV_jet = uproot_methods.TLorentzVectorArray(
              jetAK8_px,
              jetAK8_py,
              jetAK8_pz,
              jetAK8_E )    
    
    W_mass = ( TLV_lep + TLV_jet ).mass # Massa invariante do WW
    W_lep_pt = ( TLV_lep ).pt # Pt do par de lepton
    
    dphi_jet_lep = TLV_lep.phi - TLV_jet.phi
    dphi_jet_lep = np.where( dphi_jet_lep >=  scipy.constants.pi, dphi_jet_lep - 2*scipy.constants.pi, dphi_jet_lep)
    dphi_jet_lep = np.where( dphi_jet_lep <  - scipy.constants.pi, dphi_jet_lep + 2*scipy.constants.pi, dphi_jet_lep) # delta phi entre o jato e o par de lepton
    dphi_jet_MET = METphi - TLV_jet.phi
    dphi_jet_MET = np.where( dphi_jet_MET >=  scipy.constants.pi, dphi_jet_MET - 2*scipy.constants.pi, dphi_jet_MET)
    dphi_jet_MET = np.where( dphi_jet_MET <  - scipy.constants.pi, dphi_jet_MET + 2*scipy.constants.pi, dphi_jet_MET) # delta phi entre o jato e o MET 
    


    '''
    ** numeração das colunas do numpy array ** ( para facilitar na hora de fazer os cortes )

    0  --> massa do WW
    1  --> Pt do W leptônico
    2  --> DeltaPhi entre W_hadrônico e W_leptônico
    3  --> DeltaPhi entre Jatos e o MET
    4  --> jetAK8_pt
    5  --> jetAK8_eta
    6  --> jetAK8_prunedMass
    7  --> jetAK8_tau21
    8  --> METPt
    9  --> muon_pt
    10 --> muon_eta
    11 --> ExtraTracks

    '''

    pfeta = get_PfFrom(tree, 'pfeta')
    pfphi = get_PfFrom(tree, 'pfphi') 
    pffromPV = pd.DataFrame(get_PfFrom(tree, 'pffromPV'))

    dR_muon = pd.DataFrame( ( ( ( pfeta - muon_eta )**2 + ( pfphi - muon_phi )**2 )**0.5) )
    dR_jet =  pd.DataFrame( ( ( (pfeta - TLV_jet.eta )**2 + ( pfphi - TLV_jet.phi )**2 )**0.5) )

    pfCands_sel1_ = pffromPV[ pffromPV == 3.0 ] 
                    
    pfCands_sel2_ = pfCands_sel1_[
                    dR_muon > 0.3 
                    ] 

    pfCands_sel3_ = pfCands_sel2_[
                    dR_jet > 0.8
                    ] 
                    
    ExtraTracks = np.array( (pfCands_sel3_.T).count() )

    #print('ExtraTracks --> \n', ExtraTracks,'\n')

    events_all = np.concatenate( ( W_mass.reshape(-1,1), W_lep_pt.reshape(-1,1), dphi_jet_lep.reshape(-1,1), 
    dphi_jet_MET.reshape(-1,1),jetAK8_pt.reshape(-1,1), jetAK8_eta.reshape(-1,1), jetAK8_prunedMass.reshape(-1,1), jetAK8_tau21.reshape(-1,1), 
    METPt.reshape(-1,1), muon_pt.reshape(-1,1), muon_eta.reshape(-1,1), ExtraTracks.reshape(-1,1) ) , axis = 1 ) # concatenando todos as variáveis

    events_all_cut = (events_all[:,4] >= 200) & (events_all[:,5] <= 2.4) & (events_all[:,8] >= 40)  & (events_all[:,9] >= 53)  & (events_all[:,10] <= 2.4)  # realizando os cortes nas variáveis
    
    array_numpy = events_all[events_all_cut]

    columns = ['Mww','Pt_W_lep','dPhi_Whad_Wlep','dPhi_jatos_MET','jetAK8_pt','jetAK8_eta','jetAK8_prunedMass','jetAK8_tau21','METPt','muon_pt','muon_eta','ExtraTracks']

    DataFrame = pd.DataFrame( array_numpy , columns = columns )

    return DataFrame


def graph(lista,bins,label,fontsize_leg,xmin,xmax,xlabel,ylabel,fontsize_xlabel,fontsize_ylabel,loc_leg,lista_norm ):
    plt.hist( lista, bins = bins, stacked = True, histtype = 'stepfilled', label = label, density = False, weights = lista_norm )
    plt.legend( loc = loc_leg , fontsize=fontsize_leg )
    plt.xlim( xmin , xmax )
    plt.ylabel( ylabel , fontsize = fontsize_ylabel )
    plt.xlabel( xlabel , fontsize = fontsize_xlabel )
    plt.yscale( 'log' )
    hep.cms.label( llabel="Preliminary", rlabel="$9.792 fb^{-1}$" ) 
    #plt.savefig(PATH_PLOT+'/{}.pdf'.format(name))
    plt.show() 


def plot(lista_1,lista_2,bins0,bins1,label0,label1,fontsize_leg,xmin,xmax,xlabel,ylabel,fontsize_xlabel,fontsize_ylabel,loc_leg,lista_norm1,lista_norm2):
    fig, axes = plt.subplots( 1, 2, figsize=(10,10) )
    axes[0].hist( lista_1, bins = bins0, stacked=False, histtype = 'step', label=label0, density = False, weights = lista_norm1, color = ['cyan', 'green', 'red', 'fuchsia','gold'] )
    axes[0].legend(loc=loc_leg, fontsize=fontsize_leg)
    axes[0].set_xlim(xmin,xmax)
    axes[0].set_ylabel(ylabel, fontsize = fontsize_ylabel)
    axes[0].set_yscale('log')
    axes[0] = hep.cms.cmslabel(data=False, paper=False, year='$9.792 fb^{-1}$', ax = axes[0])

    axes[1].hist( lista_2, bins = bins1, stacked=False, histtype = 'step', label=label1, density = False, weights = lista_norm2, color = ['cyan', 'green', 'red', 'fuchsia','gold'] )
    axes[1].legend(loc=loc_leg, fontsize=fontsize_leg)
    axes[1].set_xlim(xmin,xmax)
    axes[1].set_xlabel(xlabel,fontsize = fontsize_xlabel)
    axes[1].set_yscale('log')
    axes[1] = hep.cms.cmslabel(data=False, paper=False, year='$9.792 fb^{-1}$', ax = axes[1])    
    #plt.savefig(PATH_PLOT+'/{}.pdf'.format(name))
    plt.show()            