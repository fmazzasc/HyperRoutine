## script to get the integrated yield of the hypertriton spectrum
import ROOT
import numpy as np
import yaml
import argparse

## levy-tsallis is defined in the file AdditionalFunctions.h
ROOT.gROOT.SetBatch(True)
ROOT.gROOT.ProcessLine('.L utils/AdditionalFunctions.h++')
from ROOT import LevyTsallis

parser = argparse.ArgumentParser(description='Configure the parameters of the script.')
parser.add_argument('--config-file', dest='config_file', help="path to the YAML file with configuration.", default='')
args = parser.parse_args()
if args.config_file == "":
    print('** No config file provided. Exiting. **')
    exit()


config_file = open(args.config_file, 'r')
config = yaml.full_load(config_file)

input_file = ROOT.TFile(config['input_file_spectrum'])
h3l_spectrum = input_file.Get('std/hStat')
h3l_spectrum_syst = input_file.Get('std/hSyst')
h3l_spectrum.SetDirectory(0)

## comparison with tommca model
tommca_file = ROOT.TFile('utils/models/HypertritonToMCCA_gaus.root')
tommca_pred = tommca_file.Get('MinBiasSpectrumCongleton')
tommca_pred_gaus = tommca_file.Get('MinBiasSpectrumGaussian')
tommca_pred.SetDirectory(0)
tommca_pred_gaus.SetDirectory(0)

h3l_mass = 2.99131
mt_expo = ROOT.TF1('mtexpo', '[2]*x*exp(-TMath::Sqrt(([0]*[0]+x*x))/[1])', 0., 6)
mt_expo.FixParameter(0, h3l_mass)
mt_expo.SetParLimits(1, 0.1, 1)
mt_expo.SetParLimits(2, 1.e-08, 1)
pt_expo = ROOT.TF1('ptexpo', '[1]*x*exp(-x/[0])', 0., 6)
pt_expo.SetParLimits(0, 0.1, 1)
pt_expo.SetParLimits(1, 1.e-08, 1)
levy = LevyTsallis('levy', h3l_mass)
levy.SetParLimits(1, 10, 30)
levy.SetParLimits(3, 1e-08, 2.5e-08)
## fit the spectrum with all the functions and get the integral of the fit functions
h3l_spectrum.Fit(mt_expo, 'R')
print('mt_expo integral: ', mt_expo.Integral(0., 6.), ' +/- ', mt_expo.IntegralError(0., 6.))
h3l_spectrum.Fit(pt_expo, 'R')
print('pt_expo integral: ', pt_expo.Integral(0., 6.), ' +/- ', pt_expo.IntegralError(0., 6.))
h3l_spectrum.Fit(levy, 'R')
print('levy integral: ', levy.Integral(0., 6.), ' +/- ', levy.IntegralError(0., 6.))
## plot all the fit functions and the datapoint into a single canvas
canvas = ROOT.TCanvas('canvas', 'canvas', 800, 600)
canvas.SetLogy()
## draw a new frame
canvas.DrawFrame(0, h3l_spectrum.GetMinimum(), 6, h3l_spectrum.GetMaximum()*10, f';{h3l_spectrum.GetXaxis().GetTitle()};{h3l_spectrum.GetYaxis().GetTitle()}')
## set x-axis range from 0 to 6
h3l_spectrum.GetXaxis().SetRangeUser(0., 6.)
h3l_spectrum.SetMarkerStyle(20)
h3l_spectrum.SetMarkerSize(0.5)
h3l_spectrum.SetMarkerColor(ROOT.kBlack)
h3l_spectrum.SetLineColor(ROOT.kBlack)
## remove fit function attached to the histogram
h3l_spectrum.SetStats(0)
h3l_spectrum.GetListOfFunctions().Delete()
h3l_spectrum.Draw("same")
mt_expo.SetLineColor(ROOT.kRed)
mt_expo.SetLineWidth(2)
mt_expo.Draw('same')
pt_expo.SetLineColor(ROOT.kBlue)
pt_expo.SetLineWidth(2)
pt_expo.Draw('same')
levy.SetLineColor(ROOT.kGreen)
levy.SetLineWidth(2)
levy.Draw('same')


## connected errorband using a TGraphErrors
gr = ROOT.TGraphErrors(tommca_pred)
gr.SetMarkerSize(0)
gr.SetLineWidth(2)
gr.SetLineColor(ROOT.kRed)
gr.SetMarkerColor(ROOT.kRed)
gr.SetFillColorAlpha(ROOT.kRed, 0.5)
gr_gaus = ROOT.TGraphErrors(tommca_pred_gaus)
gr_gaus.SetMarkerSize(0)
gr_gaus.SetLineWidth(2)
gr_gaus.SetLineColor(ROOT.kOrange)
gr_gaus.SetMarkerColor(ROOT.kOrange)
gr_gaus.SetFillColorAlpha(ROOT.kOrange, 0.5)
# plot tomcca predictions as a shaded red error band
pinfo_alice = ROOT.TPaveText(0.53, 0.6, 0.89, 0.85, 'NDC')
pinfo_alice.SetBorderSize(0)
pinfo_alice.SetFillStyle(0)
pinfo_alice.SetTextAlign(11)
pinfo_alice.SetTextFont(42)
pinfo_alice.AddText('ALICE Preliminary')
pinfo_alice.AddText('Run 3 pp,#kern[0.4]{#sqrt{#it{s}}} = 13.6 TeV')
pinfo_alice.AddText('#it{L}_{int} = 63 pb^{-1}')
pinfo_alice.AddText('#pm 10% global unc. not shown')

c = ROOT.TCanvas('c', 'c', 800, 600)
## increase left margin to make room for the y-axis label
ROOT.gPad.SetLeftMargin(0.15)
frame = c.DrawFrame(0.2, 0.6e-10, 6, 2.1e-08, r';#it{p}_{T} (GeV/#it{c});#frac{1}{#it{N}_{evt}}#frac{d^{2}#it{N}}{d#it{y}d#it{p}_{T}} (GeV/#it{c})^{-1}')
## set title y offset
frame.GetYaxis().SetTitleOffset(1.3)
c.SetLogy()
gr.Draw('3 same')
gr_gaus.Draw('3 same')
h3l_spectrum.Draw('PEX0 SAME')
h3l_spectrum_syst.Draw('PE2 SAME')
h3l_spectrum.GetListOfFunctions().Clear()
h3l_spectrum_syst.GetListOfFunctions().Clear()
pinfo_alice.Draw()
leg_data = ROOT.TLegend(0.19, 0.36, 0.6, 0.46)
leg_data.SetFillStyle(0)
leg_data.SetBorderSize(0)
leg_data.SetTextFont(42)
leg_data.SetMargin(0.1)
## fix text size
leg_data.SetTextSize(0.037)
leg_data.AddEntry(h3l_spectrum, '{}^{3}_{#bar{#Lambda}}#bar{H}#kern[0.3]{#rightarrow}#kern[0.1]{^{3}#bar{He}}#kern[0.3]{+}#kern[0.3]{#pi^{+}}, |#it{y}| < 1', 'PE')
leg = ROOT.TLegend(0.19, 0.15, 0.65, 0.36)
leg.SetFillStyle(0)
leg.SetBorderSize(0)
leg.SetTextFont(42)
leg.SetMargin(0.1)
leg.SetTextSize(0.037)
leg.SetHeader('	arXiv:2504.02491 (2025)')
leg.AddEntry(gr, 'Congleton Wave Function', 'F')
leg.AddEntry(gr_gaus, 'd#minus#Lambda Gaussian Wave Function', 'F')
leg_data.Draw()
leg.Draw()

outfile = ROOT.TFile(config['output_file'], 'RECREATE')
h3l_spectrum.Write('hStat')
h3l_spectrum_syst.Write('hSyst')
mt_expo.Write('mt_expo')
pt_expo.Write('pt_expo')
levy.Write('levy')
canvas.Write('canvas')
c.Write('canvas_pt')
