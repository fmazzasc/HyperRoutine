import ROOT


from hipe4ml.tree_handler import TreeHandler
from hipe4ml.model_handler import ModelHandler
import hipe4ml.analysis_utils as au
import hipe4ml.plot_utils as pu
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('PDF')

from signal_extraction import SignalExtraction



from sklearn.utils import shuffle
from itertools import product
import copy
import yaml
import argparse
import uproot
import pandas as pd
import numpy as np
import os
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setSilentMode(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)
kOrangeC  = ROOT.TColor.GetColor('#ff7f00')

import sys
sys.path.append('utils')
import utils as utils



parser = argparse.ArgumentParser(description='Configure the parameters of the script.')
parser.add_argument('--config-file', dest='config_file', help="path to the YAML file with configuration.", default='')
args = parser.parse_args()
if args.config_file == "":
    print('** No config file provided. Exiting. **')
    exit()

config_file = open(args.config_file, 'r')
config = yaml.full_load(config_file)

## input and output files
input_file_name_data = config['input_files_data']
input_file_name_mc = config['input_files_mc']
input_analysis_results_file = config['input_analysis_results_file']
input_file_tomcca = config['input_file_tomcca']

preselections = config['preselections']
models_dir = config['models_dir']

is_trigger = config['is_trigger']
output_dir_name = config['output_dir']
output_file_suffix = config['output_file_suffix']


## general settings
calibrate_he_momentum = config['calibrate_he_momentum']
coal_based_mc = config['coal_based_mc']
pt_bins = config['pt_bins']
bin_structure = config['bin_structure']
is_matter = config['is_matter']
matter_options = ['matter', 'antimatter', 'both']
if is_matter not in matter_options:
    raise ValueError(f'Invalid is-matter option. Expected one of: {matter_options}')

## yield settings
absorption_histo_file = config['absorption_histo_file']
event_loss = config['event_loss']
signal_loss = config['signal_loss']

print('**********************************')
print('    Running pt_analysis_bdt.py')
print('**********************************\n')
print("----------------------------------")
print("** Loading data and apply preselections **")

tree_names = ['O2datahypcands','O2hypcands', 'O2hypcandsflow']
tree_keys = uproot.open(input_file_name_data[0]).keys()
for tree in tree_names:
    for key in tree_keys:
        if tree in key:
            tree_name = tree
            break
print(f"Data tree found: {tree_name}")
data_hdl = TreeHandler(input_file_name_data, tree_name, folder_name='DF*')
mc_hdl = TreeHandler(input_file_name_mc, 'O2mchypcands', folder_name='DF*')

# Add columns to the handlers
utils.correct_and_convert_df(data_hdl, calibrate_he3_pt=calibrate_he_momentum, isMC=False)
utils.correct_and_convert_df(mc_hdl, calibrate_he3_pt=calibrate_he_momentum, isMC=True)

# apply preselections + get absorption histo
matter_sel = ''
mc_matter_sel = ''
absorption_histo = None

if absorption_histo_file != '':
    absorption_file = ROOT.TFile.Open(absorption_histo_file)
    absorption_histo_mat = absorption_file.Get('x1.5/h_abso_frac_pt_mat')
    absorption_histo_anti = absorption_file.Get('x1.5/h_abso_frac_pt_antimat')
    absorption_histo_mat.SetDirectory(0)
    absorption_histo_anti.SetDirectory(0)

if is_matter == 'matter':
    matter_sel = 'fIsMatter == True'
    mc_matter_sel = 'fGenPt > 0'
    if absorption_histo_file != '':
        absorption_histo = absorption_histo_mat

elif is_matter == 'antimatter':
    matter_sel = 'fIsMatter == False'
    mc_matter_sel = 'fGenPt < 0'
    if absorption_histo_file != '':
        absorption_histo = absorption_histo_anti

if matter_sel != '':
    data_hdl.apply_preselections(matter_sel)
    mc_hdl.apply_preselections(mc_matter_sel)
    if absorption_histo_file != '':      ## get average between matter and antimatter absorption
        absorption_histo = absorption_histo_mat.Clone('h_abso_frac_pt')
        absorption_histo.Add(absorption_histo_anti)
        absorption_histo.Scale(0.5)

# reweight MC pT spectrum
spectra_file = ROOT.TFile.Open('utils/heliumSpectraMB.root')
he3_spectrum = spectra_file.Get('fCombineHeliumSpecLevyFit_0-100')
spectra_file.Close()
utils.reweight_pt_spectrum(mc_hdl, 'fAbsGenPt', he3_spectrum)
if not coal_based_mc:
    mc_hdl.apply_preselections('rej==True')
    mc_reco_hdl = mc_hdl.apply_preselections('fIsReco == 1', inplace=False)
    mc_hdl_evsel = mc_hdl.apply_preselections('fIsSurvEvSel==True', inplace=False)
else:
    signal_loss = 1
    mc_reco_hdl = mc_hdl.apply_preselections('fIsReco == 1', inplace=False)
    mc_hdl_evsel = mc_hdl

preselections = utils.convert_sel_to_string(preselections)
data_hdl.apply_preselections(preselections)
mc_reco_hdl.apply_preselections(preselections)




tomcca_file = ROOT.TFile.Open(input_file_tomcca)
tomcca_hist = tomcca_file.Get('MinBiasSpectrumCongletonFine')

output_file = ROOT.TFile.Open(output_dir_name + f"/sigscan_{output_file_suffix}.root", "RECREATE")
print("** Proceeding to application **")
h_gen_pt = ROOT.TH1D('h_gen_pt', ';p_{T} (GeV/c);Counts', len(pt_bins) - 1, np.array(pt_bins))
h_reco_pt = ROOT.TH1D('h_reco_pt', ';p_{T} (GeV/c);Counts', len(pt_bins) - 1, np.array(pt_bins))
utils.fill_th1_hist(h_gen_pt, mc_hdl_evsel, 'fAbsGenPt')
utils.fill_th1_hist(h_reco_pt, mc_reco_hdl, 'fPt')
presel_eff_hist = utils.computeEfficiency(h_gen_pt, h_reco_pt, 'PreselEff')
n_events = utils.getNEvents(input_analysis_results_file, is_trigger)
br = 0.25
delta_rap = 2.0


for ibin in range(len(pt_bins) - 1):
    pt_dir = output_file.mkdir(f'pt_bin_{ibin}')
    ptbin = [pt_bins[ibin], pt_bins[ibin + 1]]
    for bin_struct in bin_structure:
        sel_bins = bin_struct['pt_bins']
        if ptbin in sel_bins:
            break
    output_file.cd(f'pt_bin_{ibin}')
    bdt_eff_array = np.load(models_dir + f"/efficiency_arr_ptbin_{ibin}.npy")
    bdt_score_eff_array = np.load(models_dir + f"/score_efficiency_arr_ptbin_{ibin}.npy")
    pt_range = f'(fPt >= {ptbin[0]}) & (fPt < {ptbin[1]})'
    data_hdl_pt = data_hdl.apply_preselections(pt_range, inplace=False)
    mc_hdl_pt = mc_reco_hdl.apply_preselections(pt_range, inplace=False)
    model_hdl = ModelHandler()
    model_hdl.load_model_handler(models_dir + f"/model_handler_ptbin_{ibin}.pkl")
    data_hdl_pt.apply_model_handler(model_hdl, column_name="BDT_score")
    mc_hdl_pt.apply_model_handler(model_hdl, column_name="BDT_score")

    h_significance_bdt = ROOT.TH1D(f'h_significance_bdt_ptbin_{ibin}', ';BDT Efficiency;Significance', 51, 0.495, 1.005)
     ## loop over bdt efficiencies
    for bdt_eff, score in zip(bdt_eff_array, bdt_score_eff_array):
        data_hdl_eff = data_hdl_pt.apply_preselections(f'BDT_score >= {score}', inplace=False)
        mc_hdl_eff = mc_hdl_pt.apply_preselections(f'BDT_score >= {score}', inplace=False)
        
        ## fit background in the sidebands to get sigma range mc to data
        bkg_fit_func = 'pol2'
        h_bkg = ROOT.TH1D(f'h_bkg_{ibin}_bdt{int(bdt_eff*100)}', ';m_{#Lambda^{3}H} (GeV/c^{2});Counts', 30, 2.96, 3.04)
        ## convert into a tgraph for fitting and ignore non sideband regions
        utils.fill_th1_hist(h_bkg, data_hdl_eff, 'fMassH3L')
        tgraph_bkg = ROOT.TGraphErrors()
        tgraph_bkg.SetName(f'tgraph_bkg_{ibin}_bdt{int(bdt_eff*100)}')
        for bin_idx in range(1, h_bkg.GetNbinsX() + 1):
            bin_content = h_bkg.GetBinContent(bin_idx)
            bin_error = h_bkg.GetBinError(bin_idx)
            x = h_bkg.GetBinCenter(bin_idx)
            if x >= 2.98 and x <= 3.005:
                continue
            if bin_content == 0:
                continue
            y = bin_content
            ex = h_bkg.GetBinWidth(bin_idx) / 2
            ey = bin_error
            tgraph_bkg.SetPoint(tgraph_bkg.GetN(), x, y)
            tgraph_bkg.SetPointError(tgraph_bkg.GetN() - 1, ex, ey)

        bkg_fit_result = tgraph_bkg.Fit(bkg_fit_func, 'QS')
        ## get integral of fit function in the signal region
        signal_region = (2.975, 3.02)
        bkg_tf1 = tgraph_bkg.GetFunction(bkg_fit_func)
        bkg_integral = bkg_tf1.Integral(signal_region[0], signal_region[1]) / h_bkg.GetBinWidth(1)
        tgraph_bkg.Write()

        if bkg_integral < 0:
            bkg_integral = 0
                ## generate a signal according to tommcca
        signal_integral = tomcca_hist.Integral(tomcca_hist.FindBin(ptbin[0]), tomcca_hist.FindBin(ptbin[1]) - 1) * tomcca_hist.GetBinWidth(1)
        presel_eff = presel_eff_hist.GetBinContent(presel_eff_hist.FindBin((ptbin[0] + ptbin[1]) / 2))
        exp_counts = signal_integral * n_events * br * delta_rap * presel_eff * bdt_eff * signal_loss / event_loss
        print('----------------------------------')
        print(f'Expected counts for tomcaa in pt bin {ptbin} at BDT eff {bdt_eff}: {exp_counts}')
        print(f'Background integral in signal region for pt bin {ptbin} at BDT eff {bdt_eff}: {bkg_integral}')
        signficance = exp_counts / np.sqrt(exp_counts + bkg_integral) if (exp_counts + bkg_integral) > 0 else 0
        h_significance_bdt.SetBinContent(h_significance_bdt.FindBin(bdt_eff), signficance)

    output_file.cd()
    h_significance_bdt.Write()

        




