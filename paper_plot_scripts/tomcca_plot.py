import ROOT
import numpy as np
ROOT.gStyle.SetCanvasPreferGL(1)
ROOT.gROOT.SetBatch(1)

spec_file = ROOT.TFile('/home/fmazzasc/run3/results/2024_bdt/final_yield_pp2024_sigscan.root')
spectrum_stat = spec_file.Get('h_default_spectrum_stat')
spectrum_stat.SetDirectory(0)
spectrum_syst = spec_file.Get('h_default_spectrum_syst')
spectrum_syst.SetDirectory(0)

tomcca_file = ROOT.TFile('../utils/models/tomcca_congl_new.root')
tomcca_pred_up = tomcca_file.Get('ptLH3_GaussCongle_142')
tomcca_pred = tomcca_file.Get('ptLH3_GaussCongle_84')

tomcca_pred_gaus_up = tomcca_file.Get('ptLH3_ArgCongle_142')
tomcca_pred_gaus = tomcca_file.Get('ptLH3_ArgCongle_84')

tomcca_pred.SetDirectory(0)
tomcca_pred_gaus.SetDirectory(0)
tomcca_pred_up.SetDirectory(0)
tomcca_pred_gaus_up.SetDirectory(0)

## connected errorband with up and down th1ds 
gr = ROOT.TGraphErrors(tomcca_pred)
gr_up = ROOT.TGraphErrors(tomcca_pred_up)
gr_gaus = ROOT.TGraphErrors(tomcca_pred_gaus)
gr_gaus_up = ROOT.TGraphErrors(tomcca_pred_gaus_up)
for i in range(gr.GetN()):
    y = gr.GetY()[i]
    y_up = gr_up.GetY()[i]
    y_gaus = gr_gaus.GetY()[i]
    y_gaus_up = gr_gaus_up.GetY()[i]
    err_y = abs(y_up - y)
    err_y_gaus = abs(y_gaus_up - y_gaus)
    gr.SetPoint(i, (gr.GetX()[i] + gr_up.GetX()[i]) / 2, y)
    gr_gaus.SetPoint(i, (gr_gaus.GetX()[i] + gr_gaus_up.GetX()[i]) / 2, y_gaus)
    gr.SetPointError(i, 0, err_y)
    gr_gaus.SetPointError(i, 0, err_y_gaus)

gr.SetFillColorAlpha(ROOT.kRed, 0.5)
gr.SetLineColor(ROOT.kRed)

gr_gaus.SetFillColorAlpha(ROOT.kOrange, 0.5)
gr_gaus.SetLineColor(ROOT.kOrange)




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
leg.AddEntry(gr, 'Congleton + AV', 'F')
leg.AddEntry(gr_gaus, 'Congleton + Gaus (HWH)', 'F')
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



output = ROOT.TFile('spectra_inel.root', 'RECREATE')
output.cd()
spectrum_stat.Write()
spectrum_syst.Write()
c.Write()
output.Close()


## get integrated yield from tomcca: multiply by x errors and sum
n_ev = 0
n_ev_gaus = 0
for i in range(gr.GetN()):
    n_ev += gr.GetY()[i] * 2 * gr.GetErrorX(i)
    n_ev_gaus += gr_gaus.GetY()[i] * 2 * gr_gaus.GetErrorX(i)
print("TomCCA integrated yield:", n_ev)
print("TomCCA (gaus) integrated yield:", n_ev_gaus)