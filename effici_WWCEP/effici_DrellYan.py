# -*- coding: utf-8 -*-

import h5py
import numpy as np
import pandas as pd

PATH = "/eos/home-m/matheus/output_AQAg/" # Caminho comum para todos os arquivos .h5

def open_file( file ):
    df = None
    with h5py.File( file , 'r' ) as f:
        dset = f['dados']
        #print ( 'antes do corte nos xis:', dset.shape )
        #print ( dset[:,:] )
        array = np.array( dset )
        #array_cut = (array[:,0] > 600) & (array[:,1] > 200) & (array[:,2] > 2) & (array[:,3] > 2) & (array[:,4] > 200) & (array[:,5] < 2.4) & (array[:,7] < 0.6) & (array[:,8] > 40) & (array[:,9] > 53)  & (array[:,10] < 2.4) # Corte no xi1 e no xi2
        array_cut = (array[:,5] < 2.4) & (array[:,10] < 2.4) & (array[:,8] > 40) & (array[:,9] > 53) & (array[:,4] > 200)
        DataSet_ = array[array_cut]
        arrau_cut_xi = (DataSet_[:,15] > 0.04) & (DataSet_[:,16] > 0.04) & (DataSet_[:,15] < 0.111) & (DataSet_[:,16] < 0.138)        
        DataSet_ = DataSet_[arrau_cut_xi]
        #DataSet_ = array
        Mx = 13000 * ( np.sqrt( DataSet_[:,15] * DataSet_[:,16] ) )
        Yx = 0.5 * ( np.log( DataSet_[:,16] / DataSet_[:,15] ) )
        Mww_Mxx =  DataSet_[:,0] / Mx 
        Yww_Yx = DataSet_[:,13] - Yx
        DataSet = np.concatenate( ( DataSet_, Mx.reshape(-1,1), Yx.reshape(-1,1), Mww_Mxx.reshape(-1,1), Yww_Yx.reshape(-1,1) ), axis = 1 )        
        #mask = np.any( np.isnan( DataSet ) , axis = 1 )
        #print(DataSet.shape)
        print( 'Depois do corte nos xis:', DataSet.shape)
        MultiRP = ( DataSet[:,25] == 1 ) & ( DataSet[:,26] == 1 )
        print( 'MultiRP Events :: ', DataSet[ MultiRP ].shape )
        return DataSet[ MultiRP ]
        #return DataSet

SingleMuon_Run2016B = 4.55
SingleMuon_Run2016C = 1.59
SingleMuon_Run2016G = 3.65
Luminosidade        = SingleMuon_Run2016B + SingleMuon_Run2016C + SingleMuon_Run2016G 


import ROOT
pwd_effi = '/eos/home-m/matheus/SWAN_projects/Acoplamento_Quartico_Anomalo/'

def EfficienciesStudies( DataSet ):
    
    # Muon Efficiency
    f_SF_ID_BC = ROOT.TFile.Open(pwd_effi + "EfficienciesStudies_2016_legacy_rereco_rootfiles_RunBCDEF_SF_ID.root")
    f_SF_ISO_BC = ROOT.TFile.Open(pwd_effi +  "EfficienciesStudies_2016_legacy_rereco_rootfiles_RunBCDEF_SF_ISO.root" )
    f_SF_trg_BC = ROOT.TFile.Open(pwd_effi +  "EfficienciesAndSF_trg_RunBtoF.root" )
    f_SF_ID_G = ROOT.TFile.Open(pwd_effi + "EfficienciesStudies_2016_legacy_rereco_rootfiles_RunGH_SF_ID.root")
    f_SF_ISO_G = ROOT.TFile.Open(pwd_effi + "EfficienciesStudies_2016_legacy_rereco_rootfiles_RunGH_SF_ISO.root")
    f_SF_trg_G = ROOT.TFile.Open(pwd_effi + "EfficienciesAndSF_trg_RunGH.root")

    h_SF_ID_BC = f_SF_ID_BC.Get("NUM_TightID_DEN_genTracks_eta_pt")
    h_SF_ISO_BC = f_SF_ISO_BC.Get("NUM_TightRelIso_DEN_TightIDandIPCut_eta_pt")
    h_SF_TRG_BC = f_SF_trg_BC.Get("IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio")
    h_SF_ID_G = f_SF_ID_G.Get("NUM_TightID_DEN_genTracks_eta_pt")
    h_SF_ISO_G = f_SF_ISO_G.Get("NUM_TightRelIso_DEN_TightIDandIPCut_eta_pt")
    h_SF_TRG_G = f_SF_trg_G.Get("IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio")

    # Extra Tracks Reweight
    f_track_re = ROOT.TFile.Open(pwd_effi +  "Extra_track_reweight.root" )
    
    h_track_re = f_track_re.Get( "h5" )
    
    weigth_id_BC  = [] # For muon 
    weigth_iso_BC = [] # For muon
    weigth_trg_BC = [] # For muon
    weight_id_G = []   # For muon
    weight_iso_G = []  # For muon
    weight_trg_G = []  # For muon
    
    weigth_track_re = [] # For Extra Tracks
    weigth_track_re_ele = [] # For Extra Tracks


    for i in range( 0, len( DataSet ) ):
        if [DataSet[:,9] < 120][0][i]:
            weigth_id_BC.append( h_SF_ID_BC.GetBinContent( h_SF_ID_BC.GetXaxis().FindBin( DataSet[:,10][i] ), h_SF_ISO_BC.GetYaxis().FindBin( DataSet[:,9][i] ) ) )
        else:
            weigth_id_BC.append( h_SF_ID_BC.GetBinContent( h_SF_ID_BC.GetXaxis().FindBin( DataSet[:,10][i] ), h_SF_ISO_BC.GetYaxis().FindBin( 119 ) ) )
       
    
    for i in range( 0, len( DataSet ) ):
        if [ DataSet[:,9] < 120 ][0][i]:
            weigth_iso_BC.append( h_SF_ISO_BC.GetBinContent( h_SF_ISO_BC.GetXaxis().FindBin( DataSet[:,10][i] ), h_SF_ISO_BC.GetYaxis().FindBin( DataSet[:,9][i] ) ) )
        else:
            weigth_iso_BC.append( h_SF_ISO_BC.GetBinContent( h_SF_ISO_BC.GetXaxis().FindBin( DataSet[:,10][i] ), h_SF_ISO_BC.GetYaxis().FindBin( 119 ) ) )
   
    for i in range( 0, len( DataSet ) ):
        if [ DataSet[:,9] < 120 ][0][i]:
            weight_id_G.append( h_SF_ID_G.GetBinContent( h_SF_ID_G.GetXaxis().FindBin( DataSet[:,10][i] ), h_SF_ID_G.GetYaxis().FindBin( DataSet[:,9][i] ) ) )
        else:
            weight_id_G.append( h_SF_ID_G.GetBinContent( h_SF_ID_G.GetXaxis().FindBin( DataSet[:,10][i] ), h_SF_ID_G.GetYaxis().FindBin( 119 ) ) )
    
    for i in range( 0, len( DataSet ) ):
        if [ DataSet[:,9] < 120 ][0][i]:
            weight_iso_G.append( h_SF_ISO_G.GetBinContent( h_SF_ISO_G.GetXaxis().FindBin( DataSet[:,10][i] ), h_SF_ISO_G.GetYaxis().FindBin( DataSet[:,9][i] ) ) )
        else:
            weight_iso_G.append( h_SF_ISO_G.GetBinContent( h_SF_ISO_G.GetXaxis().FindBin( DataSet[:,10][i] ), h_SF_ISO_G.GetYaxis().FindBin( 119 ) ) )

        
    for i in range( 0, len( DataSet ) ):
        if [ DataSet[:,9] < 500 ][0][i]:
            weigth_trg_BC.append( h_SF_TRG_BC.GetBinContent( h_SF_TRG_BC.GetXaxis().FindBin( DataSet[:,10][i] ), h_SF_TRG_BC.GetYaxis().FindBin( DataSet[:,9][i] ) ) )
        else:
            weigth_trg_BC.append( h_SF_TRG_BC.GetBinContent( h_SF_TRG_BC.GetXaxis().FindBin( DataSet[:,10][i] ), h_SF_TRG_BC.GetYaxis().FindBin( 499 ) ) )
    
    for i in range( 0, len( DataSet ) ):
        if [ DataSet[:,9] < 500 ][0][i]:
            weight_trg_G.append( h_SF_TRG_G.GetBinContent( h_SF_TRG_G.GetXaxis().FindBin( DataSet[:,10][i] ), h_SF_TRG_G.GetYaxis().FindBin( DataSet[:,9][i] ) ) )
        else:
            weight_trg_G.append( h_SF_TRG_G.GetBinContent( h_SF_TRG_G.GetXaxis().FindBin( DataSet[:,10][i] ), h_SF_TRG_G.GetYaxis().FindBin( 499 ) ) )
                     
        
    for i in range( 0, len( DataSet ) ):
        if [ DataSet[:,11] == 0 ][0][i]:
            weigth_track_re.append( h_track_re.GetBinContent( h_track_re.GetXaxis().FindBin( 0.1 ) ) )
        else:
            weigth_track_re.append( h_track_re.GetBinContent( h_track_re.GetXaxis().FindBin( DataSet[:,11][i] ) ) )
 
    weigth_id = np.array( weigth_id_BC )*(6.14/9.79) + np.array( weight_id_G )*(3.65/9.79)
    weigth_iso = np.array( weigth_iso_BC )*(6.14/9.79) + np.array( weight_iso_G )*(3.65/9.79)
    weigth_trg = np.array( weigth_trg_BC )*(6.14/9.79) + np.array( weight_trg_G )*(3.65/9.79)
    
    return weigth_id*weigth_iso*weigth_trg*np.array(weigth_track_re)


cross_section_TT = 831.7
cross_section_inclusive_WZ = 10.73
cross_section_inclusive_WW = 49.997
cross_section_inclusive_ZZ = 3.28
cross_section_ST_s_channel = 3.365
cross_section_ST_t_channel_top = 136.02
cross_section_ST_t_channel_antitop = 80.95
cross_section_ST_tW_top= 35.85
cross_section_ST_tW_antitop = 35.85
cross_section_DYJetsToLL_Pt_100To250 = 83.12
cross_section_DYJetsToLL_Pt_250To400 = 3.047
cross_section_DYJetsToLL_Pt_400To650 = 0.3921 
cross_section_DYJetsToLL_Pt_650ToInf = 0.0363
cross_section_QCD_Pt_170to300 = 8654. 
cross_section_QCD_Pt_300to470 = 797.3
cross_section_QCD_Pt_470to600 = 79.0
cross_section_QCD_Pt_600to800 = 25.09
cross_section_QCD_Pt_800to1000 = 4.7
cross_section_QCD_Pt_1000toInf = 1.6
cross_section_WJetsToLNu_Pt_100To250 = 677.82 
cross_section_WJetsToLNu_Pt_250To400 = 24.083
cross_section_WJetsToLNu_Pt_400To600 = 3.0563 
cross_section_WJetsToLNu_Pt_600ToInf = 0.4602


number_events_TT = 76915549

number_events_inclusive_WZ = 24311445
number_events_inclusive_WW = 6655400 + 1999200
number_events_inclusive_ZZ = 15061141 + 755866

number_events_ST_s_channel = 1000000
number_events_ST_t_channel_top = 43864048
number_events_ST_t_channel_antitop = 38811017
number_events_ST_tW_top = 6952830
number_events_ST_tW_antitop = 6933094

number_events_DYJetsToLL_Pt_100To250 =  2991815 + 2805972 + 2046961
number_events_DYJetsToLL_Pt_250To400 =  594317 + 590806 + 423976
number_events_DYJetsToLL_Pt_400To650 = 604038 + 589842 + 432056
number_events_DYJetsToLL_Pt_650ToInf = 597526 + 430691

number_events_QCD_Pt_170to300 = 19789673 + 7947159
number_events_QCD_Pt_300to470 = 24605508 + 16462878 + 7937590
number_events_QCD_Pt_470to600 = 9847664 + 5668793 + 3972819
number_events_QCD_Pt_600to800 = 9928218 + 5971175 + 401013
number_events_QCD_Pt_800to1000 = 9966149 + 6011849 + 3962749
number_events_QCD_Pt_1000toInf = 9638102 + 3990117

number_events_WJetsToLNu_Pt_100To250 = 10088599 + 9944879 
number_events_WJetsToLNu_Pt_250To400 = 10021205 + 1001250 + 1000132
number_events_WJetsToLNu_Pt_400To600 = 988234 + 951713
number_events_WJetsToLNu_Pt_600ToInf = 985127 + 989482

norm_TT = ( cross_section_TT * 1000 * Luminosidade ) / number_events_TT

norm_inclusive_WZ = ( cross_section_inclusive_WZ * 1000 * Luminosidade ) / number_events_inclusive_WZ
norm_inclusive_ZZ = ( cross_section_inclusive_ZZ * 1000 * Luminosidade ) / number_events_inclusive_ZZ
norm_inclusive_WW = ( cross_section_inclusive_WW * 1000 * Luminosidade ) / number_events_inclusive_WW

norm_ST_s_channel = ( cross_section_ST_s_channel * 1000 * Luminosidade ) / number_events_ST_s_channel
norm_ST_t_channel_top = ( cross_section_ST_t_channel_top * 1000 * Luminosidade ) / number_events_ST_t_channel_top
norm_ST_t_channel_antitop = ( cross_section_ST_t_channel_antitop * 1000 * Luminosidade) / number_events_ST_t_channel_antitop
norm_ST_tW_antitop = ( cross_section_ST_tW_antitop * Luminosidade * 1000 ) / number_events_ST_tW_antitop
norm_ST_tW_top = ( cross_section_ST_tW_top * 1000 * Luminosidade) / number_events_ST_tW_top
 
norm_QCD_Pt_170to300  = ( cross_section_QCD_Pt_170to300 * 1000  * Luminosidade ) / ( number_events_QCD_Pt_170to300  )
norm_QCD_Pt_300to470  = ( cross_section_QCD_Pt_300to470 * 1000  * Luminosidade ) / ( number_events_QCD_Pt_300to470  )
norm_QCD_Pt_470to600  = ( cross_section_QCD_Pt_470to600 * 1000  * Luminosidade ) / ( number_events_QCD_Pt_470to600  )
norm_QCD_Pt_600to800  = ( cross_section_QCD_Pt_600to800 * 1000  * Luminosidade ) / ( number_events_QCD_Pt_600to800  )
norm_QCD_Pt_800to1000 = ( cross_section_QCD_Pt_800to1000 * 1000 * Luminosidade ) / ( number_events_QCD_Pt_800to1000 )
norm_QCD_Pt_1000toInf = ( cross_section_QCD_Pt_1000toInf * 1000 * Luminosidade ) / ( number_events_QCD_Pt_1000toInf )

norm_DYJetsToLL_Pt_100To250 = ( cross_section_DYJetsToLL_Pt_100To250 * 1000 * Luminosidade ) / ( number_events_DYJetsToLL_Pt_100To250 )
norm_DYJetsToLL_Pt_250To400 = ( cross_section_DYJetsToLL_Pt_250To400 * 1000 * Luminosidade ) / ( number_events_DYJetsToLL_Pt_250To400 )
norm_DYJetsToLL_Pt_400To650 = ( cross_section_DYJetsToLL_Pt_400To650 * 1000 * Luminosidade ) / ( number_events_DYJetsToLL_Pt_400To650 )
norm_DYJetsToLL_Pt_650ToInf = ( cross_section_DYJetsToLL_Pt_650ToInf * 1000 * Luminosidade ) / ( number_events_DYJetsToLL_Pt_650ToInf )

norm_WJetsToLNu_Pt_100To250 = ( cross_section_WJetsToLNu_Pt_100To250 * 1000 * Luminosidade ) / ( number_events_WJetsToLNu_Pt_100To250 )
norm_WJetsToLNu_Pt_250To400 = ( cross_section_WJetsToLNu_Pt_250To400 * 1000 * Luminosidade ) / ( number_events_WJetsToLNu_Pt_250To400 )
norm_WJetsToLNu_Pt_400To600 = ( cross_section_WJetsToLNu_Pt_400To600 * 1000 * Luminosidade ) / ( number_events_WJetsToLNu_Pt_400To600 )
norm_WJetsToLNu_Pt_600ToInf = ( cross_section_WJetsToLNu_Pt_600ToInf * 1000 * Luminosidade ) / ( number_events_WJetsToLNu_Pt_600ToInf )

# Pt entre 100 e 250 GeV
DY_100_250_2 = PATH + 'output-DY_100_250_2.h5'
DY_100_250_3 = PATH + 'output-DY_100_250_3.h5'
DY_100_250_4 = PATH + 'output-DY_100_250_4.h5'

DY_100_250 = np.concatenate(  ( open_file( DY_100_250_2 ), 
                                open_file( DY_100_250_3 ), 
                                open_file( DY_100_250_4 ) )  , axis = 0 ) 

weight_DY_100_250 = np.array( [ norm_DYJetsToLL_Pt_100To250 ]*len( DY_100_250 ) ).reshape(-1,1)

DY_100_250 = np.concatenate( ( DY_100_250 , weight_DY_100_250 ) , axis = 1 )


# Pt entre 250 e 400 GeV
DY_250_400_2 = PATH + 'output-DY_250_400_2.h5'
DY_250_400_3 = PATH + 'output-DY_250_400_3.h5'
DY_250_400_4 = PATH + 'output-DY_250_400_4.h5'

DY_250_400 = np.concatenate(  ( open_file( DY_250_400_2 ), 
                                open_file( DY_250_400_3 ), 
                                open_file( DY_250_400_4 ) )  , axis = 0 ) 

weight_DY_250_400 = np.array( [ norm_DYJetsToLL_Pt_250To400 ]*len( DY_250_400 ) ).reshape(-1,1)

DY_250_400 = np.concatenate( ( DY_250_400 , weight_DY_250_400 ) , axis = 1 )

# Pt entre 400 e 650 GeV
DY_400_650_1 = PATH + 'output-DY_400_650_1.h5'
DY_400_650_2 = PATH + 'output-DY_400_650_2.h5'
DY_400_650_3 = PATH + 'output-DY_400_650_3.h5'

DY_400_650 = np.concatenate(  ( open_file( DY_400_650_1 ), 
                                open_file( DY_400_650_2 ), 
                                open_file( DY_400_650_3 ) )  , axis = 0 ) 

weight_DY_400_650 = np.array( [ norm_DYJetsToLL_Pt_400To650 ]*len( DY_400_650 ) ).reshape(-1,1)

DY_400_650 = np.concatenate( ( DY_400_650 , weight_DY_400_650 ) , axis = 1 )

# Pt entre 650 e infinito GeV
DY_650_INF_1 = PATH + 'output-DY_650_INF_1.h5'
DY_650_INF_2 = PATH + 'output-DY_650_INF_2.h5'

DY_650_INF = np.concatenate(  ( open_file( DY_650_INF_1 ), 
                                open_file( DY_650_INF_2 ) )  , axis = 0 ) 

weight_DY_650_INF = np.array( [ norm_DYJetsToLL_Pt_650ToInf ]*len( DY_650_INF ) ).reshape(-1,1)

DY_650_INF = np.concatenate( ( DY_650_INF , weight_DY_650_INF ) , axis = 1 )

# Concatenando todas as constribuições de Pt para o Drell-Yan
Drell_Yan = np.concatenate( ( DY_100_250 , DY_250_400 , DY_400_650 , DY_650_INF ) , axis = 0  ) 


multiRP_Drell_Yan = np.concatenate(  (Drell_Yan , EfficienciesStudies(Drell_Yan).reshape(-1,1)   ), axis = 1 )

columns_MC = ['Mww', 'Pt_W_lep', 'dPhi_Whad_Wlep', 'dPhi_jatos_MET', 'jetAK8_pt','jetAK8_eta', 'jetAK8_prunedMass',
'jetAK8_tau21', 'METPt', 'muon_pt', 'muon_eta', 'ExtraTracks', 'PUWeight', 'Yww', 'btag' ,'xi1', 'xi2', 'angulo_X_1','angulo_X_2',
'angulo_Y_1', 'angulo_Y_2', 'rpid_1', 'rpid_2', 'arm1', 'arm2', 'ismultirp_1','ismultirp_2','Mx', 'Yx', 
'Mww/Mx', 'Yww_Yx', 'Norm','weight']

DataFrame_multiRP_Drell_Yan = pd.DataFrame( multiRP_Drell_Yan, columns = columns_MC )

select_columns_MC = ['Mww', 'Pt_W_lep', 'dPhi_Whad_Wlep', 'dPhi_jatos_MET', 'jetAK8_pt','jetAK8_eta', 'jetAK8_prunedMass',
'jetAK8_tau21', 'METPt', 'muon_pt', 'muon_eta', 'ExtraTracks', 'PUWeight', 'Yww', 'btag', 'xi1', 'xi2','arm1', 'arm2', 'ismultirp_1','ismultirp_2', 'Mx', 'Yx', 
'Mww/Mx', 'Yww_Yx', 'Norm', 'weight']


with h5py.File( 'DataSet_multiRP_DrellYan.h5', 'w') as f:
   dset = f.create_dataset( 'dados', data = DataFrame_multiRP_Drell_Yan[select_columns_MC] )

print(' Salvamento Concluido ' )
