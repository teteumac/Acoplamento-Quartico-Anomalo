# -*- coding: utf-8 -*-

from __future__ import division, print_function


import uproot4
import uproot
import numpy as np
import awkward1 as ak
import pandas as pd
#import numba as nb
import scipy.constants
import uproot_methods.convert
import matplotlib.pyplot as plt
import mplhep as hep
import h5py
from math import *
import sys
import uproot_methods


def open_files( file , array_ ): # Funcao que ler e abre as trees das nTuplas
    root =  uproot4.open(file)
    tree = root['demo/Events']
    lista = []
    for events in tree.iterate( [array_] , step_size="100 MB" , library="ak" ):
        lista.append( np.array( events[ array_ ][:,0] ) )
        #print( events[ array_ ][:,0] ) 
        #merda =  np.array( pd.DataFrame( array[ array_ ].tolist() )[0] )
    #print(lista)    
    return np.concatenate( lista ).reshape(-1,1)

def open_files_PUweight( file , array_ ): # Funcao que ler e abre as trees das nTuplas
    root =  uproot4.open(file)
    tree = root['demo/Events']
    lista1 = []
    for events in tree.iterate( [array_]  , step_size = "500MB", library="pd" ):
        lista1.append( events[ array_ ] )
    return np.concatenate(lista1).reshape(-1,1)


def open_files_muon( file , array_ ):
    root_ = uproot.open( file )['demo/Events']
    merda = np.array( pd.DataFrame(root_[ array_ ].array())[0])
    return merda

def open_files_trigger( file , array_ ): # Funcao que ler e abre as trees das nTuplas
    root =  uproot4.open(file)
    tree = root['demo/Events']
    lista1 = []
    for events in tree.iterate( [array_]  , step_size = "500MB", library="pd" ):
        lista1.append( events[ array_ ][:,3] )
    return np.concatenate(lista1)

def open_files_PF( file , array_ ):
    root_ = uproot.open( file )['demo/Events']
    merda = root_[ array_ ].array()
    return merda


def open_files_protons( file , array_ ):
    root_ = uproot.open( file )['demo/Events']
    merda1 = np.array( pd.DataFrame(root_[ array_ ].array())[0])
    merda2 = np.array( pd.DataFrame(root_[ array_ ].array())[1])
    return np.concatenate( (merda1.reshape(-1,1),merda2.reshape(-1,1)),axis=1 )

def almir( file ): # Funcao que retorna um DataFrame que contem a massa invariante do WW, pt do par de le
    
    trigger =  open_files_trigger( file , 'HLT_pass') 
    
    Mw = 80.379 # massa do boson W
    
    jetAK8_pt = open_files_muon( file , 'jetAK8_pt')
    jetAK8_prunedMass = open_files_muon( file , 'jetAK8_prunedMass')
    jetAK8_tau21 = open_files_muon(file, 'jetAK8_tau21')
    jetAK8_px =  open_files_muon(file,'jetAK8_px')
    jetAK8_py =  open_files_muon(file,'jetAK8_py')
    jetAK8_pz =  open_files_muon(file,'jetAK8_pz')
    jetAK8_E =  open_files_muon(file,'jetAK8_E')
    
    print('jetAK8_py',jetAK8_py.shape)
   
    METPt  = open_files_PF(file, 'METPt' )
    METPx  = open_files_PF(file, 'METPx' )
    METPy  = open_files_PF(file, 'METPy' )
    METphi = open_files_PF(file, 'METphi')

    print( 'METPx', METPx.shape)

    muon_pt = open_files_muon(file, 'muon_pt')
    muon_eta  = open_files_muon(file, 'muon_eta')
    muon_phi  = open_files_muon(file, 'muon_phi')
    muon_px  = open_files_muon(file,'muon_px')
    muon_py  = open_files_muon(file,'muon_py')
    muon_pz  = open_files_muon(file,'muon_pz')
    muon_E  = open_files_muon(file,'muon_E')  
   
    print('muon_px',muon_px.shape) 

    jetAK4_phi = pd.DataFrame( open_files_PF(file, 'jetAK4_phi') )
    jetAK8_phi = pd.DataFrame( open_files(file, 'jetAK8_phi') )
    jetAK4_eta = pd.DataFrame( open_files_PF(file, 'jetAK4_eta') )
    jetAK8_eta = pd.DataFrame( open_files(file, 'jetAK8_eta') )
    jetAK4_btag = pd.DataFrame(open_files_PF(file,'jetAK4_btag') )

    print('jetAK4_phi',jetAK4_phi.shape)
    print('jetAK8_phi',jetAK8_phi.shape)
    print('jetAK4_btag',jetAK4_btag.shape)    
    print('muon_px',muon_px.shape)

    PUWeight = open_files_PUweight(file,'PUWeight')    
    
    k = ( ( Mw**2 ) / 2 + muon_px * METPx ) +  (muon_py * METPy ) 
    print('k',k.shape)
    raiz_ = ( ( ( (k * muon_pz)**2) / (muon_pt**4)  - ( (muon_E * METPt)**2 - k) / muon_pt**2)**0.5 )    
    raiz = np.nan_to_num(raiz_) # Os valores de NaN, causados pela divisão por 0 ou pelo resultado de uma raiz imaginária, é substituida por NaN
    Pz_nu = ( ( k * muon_pz / (muon_pt**2 ) ) + raiz ) # coordenada z do momentum do neutrino reconstruido
    W_lep_energy = ( muon_E + (METPx**2 + METPy**2 + Pz_nu**2)**0.5) # Energia do par de léptons  

    print('Pz_nu',Pz_nu.shape)

    print('W_lep_energy',W_lep_energy.shape)
    print('muon_px + METPx',(muon_px + METPx).shape)

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
    W_rapidity = (TLV_lep).rapidity # Pseudo Rapidez do W 
 
    dphi_jet_lep = TLV_lep.phi - TLV_jet.phi
    dphi_jet_lep = np.where( dphi_jet_lep >=  scipy.constants.pi, dphi_jet_lep - 2*scipy.constants.pi, dphi_jet_lep)
    dphi_jet_lep = np.where( dphi_jet_lep < -scipy.constants.pi, dphi_jet_lep + 2*scipy.constants.pi, dphi_jet_lep) # delta phi entre o jato e o par de lepton
    dphi_jet_MET = METphi - TLV_jet.phi
    dphi_jet_MET = np.where( dphi_jet_MET >=  scipy.constants.pi, dphi_jet_MET - 2*scipy.constants.pi, dphi_jet_MET)
    dphi_jet_MET = np.where( dphi_jet_MET < -scipy.constants.pi, dphi_jet_MET + 2*scipy.constants.pi, dphi_jet_MET) # delta phi entre o jato e o MET 
    
    pfeta = open_files_PF( file, 'pfeta' ) 
    pfphi = open_files_PF( file, 'pfphi' ) 
    pffromPV = open_files_PF( file, 'pffromPV' ) 
    
    
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
    ProtCand_xi = open_files_protons(file,'ProtCand_xi')
    #ProtCand_t = open_files_protons(file,'ProtCand_t')[trigger]
    ProtCand_ThX = open_files_protons(file,'ProtCand_ThX')
    ProtCand_ThY = open_files_protons(file,'ProtCand_ThY')
    ProtCand_rpid = open_files_protons(file,'ProtCand_rpid')
    ProtCand_arm = open_files_protons(file,'ProtCand_arm')
    ProtCand_ismultirp = open_files_protons(file,'ProtCand_ismultirp')

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
    12 --> PUWeight
    13 --> xi do proton 1
    14 --> xi do proton 2
    15 --> ângulo X 1
    16 --> ângulo X 2
    17 --> ângulo Y 1
    18 --> ângulo Y 2
    19 --> rpid 1
    20 --> rpid 2
    21 --> braço 1
    22 --> braço 2
    23 --> multirp 1
    24 --> multirp 2
    '''

    events_all = np.concatenate( ( W_mass.reshape(-1,1), W_lep_pt.reshape(-1,1), dphi_jet_lep.reshape(-1,1), dphi_jet_MET.reshape(-1,1), jetAK8_pt.reshape(-1,1), abs(TLV_jet.eta.reshape(-1,1)), jetAK8_prunedMass.reshape(-1,1), jetAK8_tau21.reshape(-1,1), METPt.reshape(-1,1), muon_pt.reshape(-1,1), abs(muon_eta.reshape(-1,1)), ExtraTracks , PUWeight.reshape(-1,1), W_rapidity.reshape(-1,1),BtagVeto, ProtCand_xi, ProtCand_ThX, ProtCand_ThY, ProtCand_rpid, ProtCand_arm, ProtCand_ismultirp ) , axis = 1 ) # concatenando todos as variáveis

    #events_all_datadriven = np.concatenate( ( W_mass.reshape(-1,1), W_lep_pt.reshape(-1,1), dphi_jet_lep.reshape(-1,1), dphi_jet_MET.reshape(-1,1), jetAK8_pt.reshape(-1,1), abs(TLV_jet.eta.reshape(-1,1)), jetAK8_prunedMass.reshape(-1,1), jetAK8_tau21.reshape(-1,1), METPt.reshape(-1,1), muon_pt.reshape(-1,1), abs(muon_eta.reshape(-1,1)), ExtraTracks , W_rapidity.reshape(-1,1) ) , axis = 1 )    

    events_all_trigger = events_all[trigger]
    #events_all_trigger = events_all_datadriven[trigger]
    #events_isMultiRP = (events_all[:,24] == 1) & (events_all[:,25] == 1)

    #events_all_cut = (events_all_trigger[:,4] > 200) & (events_all_trigger[:,5] < 2.4) & (events_all_trigger[:,8] > 40)  & (events_all_trigger[:,9] > 53)  & (events_all_trigger[:,10] < 2.4)  # realizando os cortes nas variáveis
    
    #array_numpy = events_all[events_all_cut]  
        
    #mask = np.any( np.isnan( events_all_trigger ) , axis = 1 )
    
    #print( pd.DataFrame( array_numpy ) )

    #return events_all_trigger[ ~mask ] # ou retorna o DataFrame com as colunas 
    return events_all_trigger


