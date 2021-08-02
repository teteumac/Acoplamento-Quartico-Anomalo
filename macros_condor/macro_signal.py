#from __future__ import division, print_function
# -*- coding: utf-8 -*-
#import uproot4
import uproot
import numpy as np
#import awkward1 as ak
import pandas as pd
import scipy.constants
import uproot_methods.convert
import matplotlib.pyplot as plt
import mplhep as hep
import h5py
from math import *
import sys
import uproot_methods
import uproot4

#def open_branches( file , array_ ): # Funcao que ler e abre as trees das nTuplas
#    root =  uproot4.open(file)
#    tree = root['demo/Events']
#    lista = []
#    for events in tree.iterate( [array_] , step_size="100 MB" , library="ak" ):
#        lista.append( np.array( events[ array_ ][:,0] ) )
        #print( events[ array_ ][:,0] ) 
        #merda =  np.array( pd.DataFrame( array[ array_ ].tolist() )[0] )
    #print(lista)    
#    return np.concatenate( lista )

def open_branches_PF( file , array_ ):
    root_ = uproot.open( file )['demo/Events']
    merda = root_[ array_ ].array()
    return merda

def open_branches_muon( file , array_ ):
    root_ = uproot.open( file )['demo/Events']
    merda = np.array( pd.DataFrame(root_[ array_ ].array())[0])
    return merda

def open_branches_protons( file , array_ ):
    root_ = uproot.open( file )['demo/Events']
    merda1 = np.array( pd.DataFrame(root_[ array_ ].array())[0])
    merda2 = np.array( pd.DataFrame(root_[ array_ ].array())[1])
    return np.concatenate( (merda1.reshape(-1,1),merda2.reshape(-1,1)),axis=1 )

def open_PUWeight( file ):
    root_ = uproot4.open( file )['demo/Events']
    return np.array( root_['PUWeight'].array() ).reshape(-1,1)


def almir( file ): # Funcao que retorna um DataFrame que contem a massa invariante do WW, pt do par de le
    
        
    Mw = 80.379 # massa do boson W
    
    # variáveis dos jatos
    jetAK8_pt = open_branches_muon( file , 'jetAK8_pt')
    jetAK8_prunedMass = open_branches_muon( file , 'jetAK8_prunedMass')
    jetAK8_tau21 = open_branches_muon(file, 'jetAK8_tau21')
    jetAK8_px =  open_branches_muon(file, 'jetAK8_px')
    jetAK8_py =  open_branches_muon(file, 'jetAK8_py')
    jetAK8_pz =  open_branches_muon(file, 'jetAK8_pz')
    jetAK8_E =  open_branches_muon(file, 'jetAK8_E')
    
    # variáveis da energia perdida
    METPt  = open_branches_PF(file, 'METPt' )
    METPx  = open_branches_PF(file, 'METPx' )
    METPy  = open_branches_PF(file, 'METPy' )
    METphi = open_branches_PF(file, 'METphi')
    
    # variáveis do muon
    muon_pt  = open_branches_muon(file, 'muon_pt')
    muon_eta = open_branches_muon(file, 'muon_eta')
    muon_phi = open_branches_muon(file, 'muon_phi')
    muon_px  = open_branches_muon(file, 'muon_px')
    muon_py  = open_branches_muon(file, 'muon_py')
    muon_pz  = open_branches_muon(file, 'muon_pz')
    muon_E   = open_branches_muon(file, 'muon_E') 
   
    jetAK4_phi = pd.DataFrame( open_branches_PF(file, 'jetAK4_phi') )
    jetAK8_phi = pd.DataFrame( open_branches_muon(file, 'jetAK8_phi') )
    jetAK4_eta = pd.DataFrame( open_branches_PF(file, 'jetAK4_eta') )
    jetAK8_eta = pd.DataFrame( open_branches_muon(file, 'jetAK8_eta') )
    jetAK4_btag = pd.DataFrame(open_branches_PF(file,'jetAK4_btag') )
   
 
    k = ( ( Mw**2 ) / 2 + muon_px * METPx ) +  (muon_py * METPy ) 
    raiz_ = ( np.sqrt( ( (k * muon_pz)**2) / (muon_pt**4)  - ( (muon_E * METPt)**2 - k) / muon_pt**2) )    
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
    
    
    print('TLV_lep + TLV_jet',(TLV_lep + TLV_jet).shape) 
    W_mass = ( TLV_lep + TLV_jet ).mass # Massa invariante do WW
    W_lep_pt = ( TLV_lep ).pt # Pt do par de lepton
    W_rapidity = (TLV_lep).rapidity # Pseudo Rapidez do W   
 
    dphi_jet_lep = TLV_lep.phi - TLV_jet.phi
    dphi_jet_lep = np.where( dphi_jet_lep >=  scipy.constants.pi, dphi_jet_lep - 2*scipy.constants.pi, dphi_jet_lep)
    dphi_jet_lep = np.where( dphi_jet_lep < -scipy.constants.pi, dphi_jet_lep + 2*scipy.constants.pi, dphi_jet_lep) # delta phi entre o jato e o par de lepton
    dphi_jet_MET = METphi - TLV_jet.phi
    dphi_jet_MET = np.where( dphi_jet_MET >=  scipy.constants.pi, dphi_jet_MET - 2*scipy.constants.pi, dphi_jet_MET)
    dphi_jet_MET = np.where( dphi_jet_MET < -scipy.constants.pi, dphi_jet_MET + 2*scipy.constants.pi, dphi_jet_MET) # delta phi entre o jato e o MET 
    
    # Calculando os Traços Extras

    pfeta = open_branches_PF( file, 'pfeta' )
    pfphi = open_branches_PF( file, 'pfphi' )
    pffromPV = open_branches_PF( file, 'pffromPV' )
    
    
    dR_muon = ( ( pfeta - muon_eta )**2 + ( pfphi - muon_phi )**2 )**0.5
    dR_jet =  ( ( pfeta - TLV_jet.eta )**2 + ( pfphi - TLV_jet.phi )**2 )**0.5

    dR_muon = dR_muon[pffromPV == 3]
    dR_jet = dR_jet[pffromPV == 3]  
    
    dR_jet = dR_jet[dR_muon > 0.3]

    dR_jet = dR_jet[dR_jet > 0.8]
    
    ExtraTracks = [len(arr) for arr in dR_jet]
    
    ExtraTracks = np.array(ExtraTracks).reshape(-1,1)

    # b-tagging

    lista_DeltaPhi = []
    lista_DeltaEta = []
    for i in range( jetAK4_phi.shape[1] ):
        delta_phi = np.array( jetAK4_phi[i] ) - np.array( jetAK8_phi[0] )
        delta_phi = np.where( delta_phi >=  scipy.constants.pi, delta_phi - 2*scipy.constants.pi, delta_phi )
        delta_phi = np.where( delta_phi < -scipy.constants.pi, delta_phi + 2*scipy.constants.pi, delta_phi )
        lista_DeltaPhi.append(delta_phi)
        lista_DeltaEta.append( np.array( jetAK4_eta[i] ) - np.array( jetAK8_eta[0] ) )

    Delta_R = np.array( np.sqrt( np.array(lista_DeltaPhi)**2 + np.array(lista_DeltaEta)**2 ) ).T
    DataFrame_DeltaR = pd.DataFrame(Delta_R)
    BtagVeto = jetAK4_btag[DataFrame_DeltaR > 0.8] > 0.9535
    BtagVeto = BtagVeto.T.sum()
    BtagVeto = np.array(BtagVeto).reshape(-1,1)
    
    # variáveis dos prótons
    ProtCand_xi = open_branches_protons(file,'ProtCand_xi')
    #ProtCand_t = open_files_protons(file,'ProtCand_t')
    ProtCand_ThX = open_branches_protons(file,'ProtCand_ThX')
    ProtCand_ThY = open_branches_protons(file,'ProtCand_ThY')
    ProtCand_rpid = open_branches_protons(file,'ProtCand_rpid')
    ProtCand_arm = open_branches_protons(file,'ProtCand_arm')
    ProtCand_ismultirp = open_branches_protons(file,'ProtCand_ismultirp')
    ProtCand_xn = open_branches_protons(file, 'ProtCand_xn')
    ProtCand_yn = open_branches_protons(file, 'ProtCand_yn')
    ProtCand_xf = open_branches_protons(file, 'ProtCand_xf')
    ProtCand_yf = open_branches_protons(file, 'ProtCand_yf')
    

    events_all = np.concatenate( ( W_mass.reshape(-1,1), W_lep_pt.reshape(-1,1), dphi_jet_lep.reshape(-1,1), dphi_jet_MET.reshape(-1,1), jetAK8_pt.reshape(-1,1), abs(TLV_jet.eta.reshape(-1,1)), jetAK8_prunedMass.reshape(-1,1), jetAK8_tau21.reshape(-1,1), METPt.reshape(-1,1), muon_pt.reshape(-1,1), abs(muon_eta.reshape(-1,1)), ExtraTracks , open_PUWeight( file ), W_rapidity.reshape(-1,1), BtagVeto, ProtCand_xi, ProtCand_ThX, ProtCand_ThY, ProtCand_rpid, ProtCand_arm, ProtCand_ismultirp, ProtCand_xn, ProtCand_yn, ProtCand_xf, ProtCand_yf) , axis = 1 ) # concatenando todos as variáveis

    events_all_cut = (events_all[:,4] > 200) & (events_all[:,5] < 2.4) & (events_all[:,8] > 40)  & (events_all[:,9] > 53)  & (events_all[:,10] < 2.4)  # realizando os cortes nas variáveis
 
    array_numpy = events_all[events_all_cut] # seleciona os eventos com base nos cortes preliminares 
    
    events_isMultiRP = ( events_all[:,24] ==1 )  & ( events_all[:,25] == 1 ) # seleciona os eventos que são MultiRP em cada braço do PPS
   

    mask = np.any( np.isnan( events_all ) , axis = 1 ) # elimina todas as linhas que contém NaN
    
    #return events_all[ ~mask ] # retorna o dataset sem os NaN
     
    return events_all # retorna o dataset com os NaN apenas com os eventos que são MultiRP


