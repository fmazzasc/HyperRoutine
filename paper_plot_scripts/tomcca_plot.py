import ROOT
import numpy as np
ROOT.gStyle.SetCanvasPreferGL(1)
ROOT.gROOT.SetBatch(1)

spec_file = ROOT.TFile('/home/fmazzasc/run3/results/2024_bdt/final_yield_pp2024_sigscan.root')
spectrum_stat = spec_file.Get('h_default_spectrum_stat')
spectrum_stat.SetDirectory(0)
spectrum_syst = spec_file.Get('h_default_spectrum_syst')
spectrum_syst.SetDirectory(0)

tomcca_file = ROOT.TFile('../utils/models/HypertritonToMCCA_gaus.root')
tomcca_pred = tomcca_file.Get('MinBiasSpectrumCongleton')
tomcca_pred_gaus = tomcca_file.Get('MinBiasSpectrumGaussian')
tomcca_pred.SetDirectory(0)
tomcca_pred_gaus.SetDirectory(0)
## connected errorband using a TGraphErrors
gr = ROOT.TGraphErrors(tomcca_pred)
gr.SetMarkerSize(0)
gr.SetLineWidth(2)
gr.SetLineColor(ROOT.kRed)
gr.SetMarkerColor(ROOT.kRed)
gr.SetFillColorAlpha(ROOT.kRed, 0.5)

gr_gaus = ROOT.TGraphErrors(tomcca_pred_gaus)
gr_gaus.SetMarkerSize(0)
gr_gaus.SetLineWidth(2)
gr_gaus.SetLineColor(ROOT.kOrange)
gr_gaus.SetMarkerColor(ROOT.kOrange)
gr_gaus.SetFillColorAlpha(ROOT.kOrange, 0.5)

## plot tomcca predictions as a shaded red error band
pinfo_alice = ROOT.TPaveText(0.53, 0.6, 0.89, 0.85, 'NDC')
pinfo_alice.SetBorderSize(0)
pinfo_alice.SetFillStyle(0)
pinfo_alice.SetTextAlign(11)
pinfo_alice.SetTextFont(42)
pinfo_alice.AddText('ALICE')
pinfo_alice.AddText('Run 3 pp,#kern[0.4]{#sqrt{#it{s}}} = 13.6 TeV')
pinfo_alice.AddText('N_{evt} = 1.9 #times 10^{12}')
pinfo_alice.AddText('#pm 10% global unc. not shown')
pinfo_alice.Draw()

c = ROOT.TCanvas('c', 'c', 800, 600)
## increase left margin to make room for the y-axis label
ROOT.gPad.SetLeftMargin(0.15)
frame = c.DrawFrame(0.2, 0.6e-10, 6, 2.1e-08, r';#it{p}_{T} (GeV/#it{c});#frac{1}{#it{N}_{evt}}#frac{d^{2}#it{N}}{d#it{y}d#it{p}_{T}} (GeV/#it{c})^{-1}')
## set title y offset
frame.GetYaxis().SetTitleOffset(1.3)
c.SetLogy()
gr.Draw('3 same')
gr_gaus.Draw('3 same')
spectrum_stat.Draw('PEX0 SAME')
spectrum_syst.Draw('PE2 SAME')
spectrum_stat.GetListOfFunctions().Clear()
spectrum_syst.GetListOfFunctions().Clear()

leg_data = ROOT.TLegend(0.19, 0.36, 0.6, 0.46)
leg_data.SetFillStyle(0)
leg_data.SetBorderSize(0)
leg_data.SetTextFont(42)
leg_data.SetMargin(0.1)
## fix text size
leg_data.SetTextSize(0.037)
leg_data.AddEntry(spectrum_stat, '{}^{3}_{#bar{#Lambda}}#bar{H}#kern[0.3]{#rightarrow}#kern[0.1]{^{3}#bar{He}}#kern[0.3]{+}#kern[0.3]{#pi^{+}}, |#it{y}| < 1', 'PE')
leg_data.Draw()


leg = ROOT.TLegend(0.19, 0.15, 0.65, 0.36)
leg.SetFillStyle(0)
leg.SetBorderSize(0)
leg.SetTextFont(42)
leg.SetMargin(0.1)
leg.SetTextSize(0.037)
leg.SetHeader('	arXiv:2504.02491 (2025)')
leg.AddEntry(gr, 'Congleton Wave Function', 'F')
leg.AddEntry(gr_gaus, 'd#minus#Lambda Gaussian Wave Function', 'F')
leg.Draw()

pinfo_alice2 = ROOT.TPaveText(0.19, 0.15, 0.5, 0.22, 'NDC')
pinfo_alice2.SetBorderSize(0)
pinfo_alice2.SetFillStyle(0)
pinfo_alice2.SetTextAlign(11)
pinfo_alice2.SetTextFont(42)
pinfo_alice.Draw()
# pinfo_alice2.Draw()
c.SaveAs('tomcca_pred.pdf')
c.SaveAs('tomcca_pred.png')
c.SaveAs('tomcca_pred.C')



h_ratio = spectrum_stat.Clone('h_ratio')
h_ratio_gaus = spectrum_stat.Clone('h_ratio_gaus')

th1_tomcca_fine = tomcca_file.Get('MinBiasSpectrumCongletonFine')
th1_tomcca_fine.SetDirectory(0)
th1_tomcca_fine_gaus = tomcca_file.Get('MinBiasSpectrumGaussianFine')
th1_tomcca_fine_gaus.SetDirectory(0)

for i in range(1, h_ratio.GetNbinsX()+1):
    ## model val taken as integral over the bin / bin width
    bin_width = h_ratio.GetXaxis().GetBinWidth(i)
    bin_low_edge = h_ratio.GetXaxis().GetBinLowEdge(i)

    val_model = 0    
    val_model_gaus = 0
    error_model = 0
    error_model_gaus = 0

    n_points = 0
    n_points_gaus = 0

    for j in range(1, th1_tomcca_fine.GetNbinsX()+1):
        tomcca_low_edge = th1_tomcca_fine.GetXaxis().GetBinLowEdge(j)
        if tomcca_low_edge >= bin_low_edge and tomcca_low_edge < bin_low_edge + bin_width:
            val_model += th1_tomcca_fine.GetBinContent(j)
            error_model += th1_tomcca_fine.GetBinError(j) * th1_tomcca_fine.GetBinError(j)
            n_points += 1
        tomcca_low_edge_gaus = th1_tomcca_fine_gaus.GetXaxis().GetBinLowEdge(j)
        if tomcca_low_edge_gaus >= bin_low_edge and tomcca_low_edge_gaus < bin_low_edge + bin_width:
            val_model_gaus += th1_tomcca_fine_gaus.GetBinContent(j)
            error_model_gaus += th1_tomcca_fine_gaus.GetBinError(j) * th1_tomcca_fine_gaus.GetBinError(j)
            n_points_gaus += 1

    val_model /= n_points
    val_model_gaus /= n_points_gaus
    model_err = np.sqrt(error_model) / n_points
    model_err_gaus = np.sqrt(error_model_gaus) / n_points_gaus
    data_val = h_ratio.GetBinContent(i)
    data_err = np.sqrt(h_ratio.GetBinError(i) * h_ratio.GetBinError(i) + spectrum_syst.GetBinError(i) * spectrum_syst.GetBinError(i))
    ratio = data_val / val_model
    ratio_gaus = data_val / val_model_gaus
    ratio_err = ratio * np.sqrt( (data_err / data_val) ** 2 + (model_err / val_model) ** 2 )
    ratio_err_gaus = ratio_gaus * np.sqrt( (data_err / data_val) ** 2 + (model_err_gaus / val_model_gaus) ** 2 )
    h_ratio.SetBinContent(i, ratio)
    h_ratio_gaus.SetBinContent(i, ratio_gaus)
    h_ratio.SetBinError(i, ratio_err)
    h_ratio_gaus.SetBinError(i, ratio_err_gaus)
    
h_ratio.SetLineColor(ROOT.kBlack)
h_ratio_gaus.SetLineColor(ROOT.kGray+2)
    


output = ROOT.TFile('spectra_inel.root', 'RECREATE')
output.cd()
spectrum_stat.Write()
spectrum_syst.Write()
c.Write()
h_ratio.Write()
h_ratio_gaus.Write()
output.Close()


## get integrated yield from tomcca: multiply by x errors and sum
n_ev = 0
for i in range(gr.GetN()):
    n_ev += gr.GetY()[i] * 2 * gr.GetErrorX(i)
print(n_ev)