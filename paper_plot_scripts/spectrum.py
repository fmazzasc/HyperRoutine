import ROOT
import numpy as np
ROOT.gStyle.SetCanvasPreferGL(1)
ROOT.gROOT.SetBatch(1)

inel_scaling = 0.75
tomcca_model_rel_unc = 0.155
use_model_bin_center_for_ratio = False
enable_levy_fit = True
h3l_mass = 2.99131


def integrate_hist_density(hist, x_min, x_max):
    total = 0.
    for i_bin in range(1, hist.GetNbinsX() + 1):
        bin_low = hist.GetBinLowEdge(i_bin)
        bin_up = bin_low + hist.GetBinWidth(i_bin)
        overlap = min(bin_up, x_max) - max(bin_low, x_min)
        if overlap > 0:
            total += hist.GetBinContent(i_bin) * overlap
    return total / (x_max - x_min) if x_max > x_min else 0.


def integrate_graph_density(graph, x_min, x_max, n_eval=200):
    if x_max <= x_min:
        return 0.
    step = (x_max - x_min) / n_eval
    total = 0.
    for i_eval in range(n_eval):
        x_val = x_min + (i_eval + 0.5) * step
        total += graph.Eval(x_val)
    return total / n_eval


def evaluate_graph_density(graph, x_min, x_max, use_bin_center=False, n_eval=200):
    if use_bin_center:
        return graph.Eval(0.5 * (x_min + x_max))
    return integrate_graph_density(graph, x_min, x_max, n_eval=n_eval)


def make_error_envelope_graphs(graph):
    graph_low = ROOT.TGraph(graph.GetN())
    graph_high = ROOT.TGraph(graph.GetN())
    for i_point in range(graph.GetN()):
        x_val = graph.GetX()[i_point]
        y_val = graph.GetY()[i_point]
        graph_low.SetPoint(i_point, x_val, y_val - graph.GetErrorYlow(i_point))
        graph_high.SetPoint(i_point, x_val, y_val + graph.GetErrorYhigh(i_point))
    return graph_low, graph_high


def build_ratio_graphs(data_stat, data_syst, model_graph, model_graph_low, model_graph_high, color):
    gr_ratio = ROOT.TGraphAsymmErrors()
    i_point = 0
    for i_bin in range(1, data_stat.GetNbinsX() + 1):
        x_low = data_stat.GetBinLowEdge(i_bin)
        x_up = x_low + data_stat.GetBinWidth(i_bin)
        x = data_stat.GetBinCenter(i_bin)
        x_err = 0.5 * data_stat.GetBinWidth(i_bin)
        model_val = evaluate_graph_density(
            model_graph,
            x_low,
            x_up,
            use_bin_center=use_model_bin_center_for_ratio,
        )
        if model_val <= 0:
            continue
        model_low_val = evaluate_graph_density(
            model_graph_low,
            x_low,
            x_up,
            use_bin_center=use_model_bin_center_for_ratio,
        )
        model_high_val = evaluate_graph_density(
            model_graph_high,
            x_low,
            x_up,
            use_bin_center=use_model_bin_center_for_ratio,
        )
        y = data_stat.GetBinContent(i_bin) / model_val
        y_data_err = np.hypot(data_stat.GetBinError(i_bin), data_syst.GetBinError(i_bin)) / model_val
        model_rel_low = max(0., model_high_val - model_val) / model_val
        model_rel_high = max(0., model_val - model_low_val) / model_val
        gr_ratio.SetPoint(i_point, x, y)
        gr_ratio.SetPointEXlow(i_point, x_err)
        gr_ratio.SetPointEXhigh(i_point, x_err)
        gr_ratio.SetPointEYlow(i_point, np.hypot(y_data_err, model_rel_low * y))
        gr_ratio.SetPointEYhigh(i_point, np.hypot(y_data_err, model_rel_high * y))
        i_point += 1

    gr_ratio.SetFillColorAlpha(color, 0.2)
    gr_ratio.SetLineColor(color)
    gr_ratio.SetLineWidth(2)

    return gr_ratio

spec_file = ROOT.TFile('/home/fmazzasc/run3/results/2024_bdt/final_yield_pp2024_sigscan_newlumi.root')
spectrum_stat = spec_file.Get('h_default_spectrum_stat')
spectrum_stat.SetDirectory(0)
spectrum_syst = spec_file.Get('h_default_spectrum_syst')
spectrum_syst.SetDirectory(0)

spectrum_stat.SetMarkerColor(ROOT.kAzure)
spectrum_stat.SetLineColor(ROOT.kAzure)
spectrum_syst.SetMarkerColor(ROOT.kAzure)
spectrum_syst.SetLineColor(ROOT.kAzure)

levy = None
if enable_levy_fit:
    ROOT.gROOT.ProcessLine('.L ../utils/AdditionalFunctions.h++')
    from ROOT import LevyTsallis

    fit_min = spectrum_stat.GetXaxis().GetBinLowEdge(1)
    fit_max = spectrum_stat.GetXaxis().GetBinUpEdge(spectrum_stat.GetNbinsX())
    levy = LevyTsallis('levy', h3l_mass)
    levy.SetParLimits(1, 5, 10)
    levy.SetParLimits(3, 1e-08, 4e-08)
    spectrum_stat.Fit(levy, 'RQ0', '', fit_min, fit_max)
    levy.SetLineColor(ROOT.kBlack)
    levy.SetLineStyle(ROOT.kDashed)
    levy.SetLineWidth(2)


tomcca_file = ROOT.TFile('../utils/output_tomcca.root')
congleton_up = tomcca_file.Get('congleton_up')
congleton_mid = tomcca_file.Get('congleton')
congleton_low = tomcca_file.Get('congleton_low')
tomcca_pred_gaus_up = tomcca_file.Get('gaussian_up')
tomcca_pred_gaus_mid = tomcca_file.Get('gaussian')
tomcca_pred_gaus_low = tomcca_file.Get('gaussian_low')
congleton_hwh = tomcca_file.Get('congleton_hwh')
congleton_up.SetDirectory(0)
congleton_mid.SetDirectory(0)
congleton_low.SetDirectory(0)
tomcca_pred_gaus_up.SetDirectory(0)
tomcca_pred_gaus_mid.SetDirectory(0)
tomcca_pred_gaus_low.SetDirectory(0)

congleton_hwh.SetDirectory(0)
congleton_hwh.Rebin(40)
congleton_hwh.Scale(1./40.)

## build asymmetric bands using the lower, mid, and upper model values
gr = ROOT.TGraphAsymmErrors(congleton_mid)
gr_gaus = ROOT.TGraphAsymmErrors(tomcca_pred_gaus_mid)
gr_hwh = ROOT.TGraphAsymmErrors(congleton_hwh)


for i in range(gr.GetN()):
    x = gr.GetX()[i]
    y_mid = gr.GetY()[i] * inel_scaling
    y_up = congleton_up.GetBinContent(i + 1) * inel_scaling
    y_low = congleton_low.GetBinContent(i + 1) * inel_scaling
    x_low = x - congleton_mid.GetBinLowEdge(i + 1)
    x_up = congleton_mid.GetBinLowEdge(i + 1) + congleton_mid.GetBinWidth(i + 1) - x
    model_unc = tomcca_model_rel_unc * y_mid
    err_low = max(0., y_mid - y_low)
    err_high = max(0., y_up - y_mid)
    gr.SetPoint(i, x, y_mid)
    gr.SetPointEXlow(i, x_low)
    gr.SetPointEXhigh(i, x_up)
    gr.SetPointEYlow(i, np.hypot(err_low, model_unc))
    gr.SetPointEYhigh(i, np.hypot(err_high, model_unc))

for i in range(gr_gaus.GetN()):
    x = gr_gaus.GetX()[i]
    y_mid = gr_gaus.GetY()[i] * inel_scaling
    y_up = tomcca_pred_gaus_up.GetBinContent(i + 1) * inel_scaling
    y_low = tomcca_pred_gaus_low.GetBinContent(i + 1) * inel_scaling
    x_low = x - tomcca_pred_gaus_mid.GetBinLowEdge(i + 1)
    x_up = tomcca_pred_gaus_mid.GetBinLowEdge(i + 1) + tomcca_pred_gaus_mid.GetBinWidth(i + 1) - x
    model_unc = tomcca_model_rel_unc * y_mid
    err_low = max(0., y_mid - y_low)
    err_high = max(0., y_up - y_mid)
    gr_gaus.SetPoint(i, x, y_mid)
    gr_gaus.SetPointEXlow(i, x_low)
    gr_gaus.SetPointEXhigh(i, x_up)
    gr_gaus.SetPointEYlow(i, np.hypot(err_low, model_unc))
    gr_gaus.SetPointEYhigh(i, np.hypot(err_high, model_unc))


for i in range(gr_hwh.GetN()):
    x = gr_hwh.GetX()[i]
    y_hwh = gr_hwh.GetY()[i] * inel_scaling
    x_low = x - congleton_hwh.GetBinLowEdge(i + 1)
    x_up = congleton_hwh.GetBinLowEdge(i + 1) + congleton_hwh.GetBinWidth(i + 1) - x
    gr_hwh.SetPoint(i, x, y_hwh)
    gr_hwh.SetPointEXlow(i, x_low)
    gr_hwh.SetPointEXhigh(i, x_up)

    err_congl_up_rel = (gr.GetErrorYhigh(i) / gr.GetY()[i]) if gr.GetY()[i] > 0 else 0
    err_congl_low_rel = (gr.GetErrorYlow(i) / gr.GetY()[i]) if gr.GetY()[i] > 0 else 0
    
    gr_hwh.SetPointEYlow(i, err_congl_low_rel * y_hwh)
    gr_hwh.SetPointEYhigh(i, err_congl_up_rel * y_hwh)

gr.SetFillColorAlpha(ROOT.kRed, 0.2)
gr.SetLineColor(ROOT.kRed)
gr.SetLineWidth(2)

gr_gaus.SetFillColorAlpha(ROOT.kAzure + 1, 0.2)
gr_gaus.SetLineColor(ROOT.kAzure + 1)
gr_gaus.SetLineWidth(2)

gr_hwh.SetFillColorAlpha(ROOT.kOrange + 7, 0.2)
gr_hwh.SetLineColor(ROOT.kOrange + 7)
gr_hwh.SetLineWidth(2)

gr_low, gr_high = make_error_envelope_graphs(gr)
gr_hwh_low, gr_hwh_high = make_error_envelope_graphs(gr_hwh)
gr_gaus_low, gr_gaus_high = make_error_envelope_graphs(gr_gaus)

ratio_congleton_band = build_ratio_graphs(
    spectrum_stat,
    spectrum_syst,
    gr,
    gr_low,
    gr_high,
    ROOT.kRed,
)
ratio_hwh_band = build_ratio_graphs(
    spectrum_stat,
    spectrum_syst,
    gr_hwh,
    gr_hwh_low,
    gr_hwh_high,
    ROOT.kOrange + 7,
)
ratio_gaus_band = build_ratio_graphs(
    spectrum_stat,
    spectrum_syst,
    gr_gaus,
    gr_gaus_low,
    gr_gaus_high,
    ROOT.kAzure + 1,
)

ratio_graphs = [
    (ratio_congleton_band, 'Congleton'),
    (ratio_hwh_band, 'Congleton H.H. tune'),
    (ratio_gaus_band, 'Double - Gaussian'),
]

ratio_max = 0.
for ratio_band, _ in ratio_graphs:
    for i_point in range(ratio_band.GetN()):
        y_val = ratio_band.GetY()[i_point]
        ratio_max = max(ratio_max, y_val + ratio_band.GetErrorYhigh(i_point))
ratio_y_max = max(2.0, 1.25 * ratio_max)




## plot tomcca predictions as asymmetric filled bands around the mid value
pinfo_alice = ROOT.TPaveText(0.6, 0.74, 0.89, 0.85, 'NDC')
pinfo_alice.SetTextSize(0.045)
pinfo_alice.SetBorderSize(0)
pinfo_alice.SetFillStyle(0)
pinfo_alice.SetTextAlign(11)
pinfo_alice.SetTextFont(42)
pinfo_alice.AddText('ALICE')
# pinfo_alice.AddText('#pm 10% global unc. not shown')
pinfo_alice.Draw()

c = ROOT.TCanvas('c', 'c', 800, 600)
## increase left margin to make room for the y-axis label
ROOT.gPad.SetLeftMargin(0.15)
frame = c.DrawFrame(0.7, 0.6e-10, 6, 2.1e-08, r';#it{p}_{T} (GeV/#it{c});#frac{1}{#it{N}_{evt}}#frac{d^{2}#it{N}}{d#it{y}d#it{p}_{T}} (GeV/#it{c})^{-1}')
## set title y offset
frame.GetYaxis().SetTitleOffset(1.3)
c.SetLogy()
gr.Draw('3 same')
gr_gaus.Draw('3 same')
gr_hwh.Draw('3 same')
if levy:
    levy.Draw('same')
spectrum_stat.Draw('PEX0 SAME')
spectrum_syst.Draw('PE2 SAME')
spectrum_stat.GetListOfFunctions().Clear()
spectrum_syst.GetListOfFunctions().Clear()
spectrum_syst.Draw('PE2 SAME')
spectrum_stat.Draw('PEX0 SAME')

leg_data = ROOT.TLegend(0.6, 0.68, 0.9, 0.77)
leg_data.SetFillStyle(0)
leg_data.SetBorderSize(0)
leg_data.SetTextFont(42)
leg_data.SetMargin(0.1)
## fix text size
leg_data.SetTextSize(0.037)
leg_data.AddEntry(spectrum_stat, 'pp, #sqrt{s} = 13.6 TeV, |#it{y}| < 1', 'PE')
if levy:
    leg_data.AddEntry(levy, 'Levy #minus Tsallis fit', 'L')
leg_data.Draw()


leg = ROOT.TLegend(0.19, 0.15, 0.65, 0.36)
leg.SetFillStyle(0)
leg.SetBorderSize(0)
leg.SetTextFont(42)
leg.SetMargin(0.1)
leg.SetTextSize(0.037)
leg.SetHeader('Coalescence (ToMCCA)')
leg.AddEntry(gr, 'Congleton', 'F')
leg.AddEntry(gr_hwh, 'Congleton H.H. tune', 'F')
leg.AddEntry(gr_gaus, 'Double #minus Gaussian', 'F')
leg.Draw()

pinfo_alice2 = ROOT.TPaveText(0.19, 0.15, 0.5, 0.22, 'NDC')
pinfo_alice2.SetBorderSize(0)
pinfo_alice2.SetFillStyle(0)
pinfo_alice2.SetTextAlign(11)
pinfo_alice2.SetTextFont(42)
pinfo_alice.Draw()
# pinfo_alice2.Draw()
c.SaveAs('spectrum_with_tomcca.pdf')
c.SaveAs('spectrum_with_tomcca.png')

c_ratio = ROOT.TCanvas('c_ratio', 'c_ratio', 800, 600)
ROOT.gPad.SetLeftMargin(0.15)
frame_ratio = c_ratio.DrawFrame(0.7, 0, 6, 3.8, r';#it{p}_{T} (GeV/#it{c});Data / model')
frame_ratio.GetYaxis().SetTitleOffset(1.3)

unity = ROOT.TLine(0.7, 1., 6, 1.)
unity.SetLineStyle(ROOT.kDashed)
unity.SetLineColor(ROOT.kGray + 2)
unity.SetLineWidth(2)
unity.Draw('same')

for ratio_band, _ in ratio_graphs:
    ratio_band.Draw('2 same')

ratio_pinfo = ROOT.TPaveText(0.18, 0.74, 0.52, 0.84, 'NDC')
ratio_pinfo.SetTextSize(0.04)
ratio_pinfo.SetBorderSize(0)
ratio_pinfo.SetFillStyle(0)
ratio_pinfo.SetTextAlign(11)
ratio_pinfo.SetTextFont(42)
ratio_pinfo.AddText('ALICE')
ratio_pinfo.AddText('pp, #sqrt{#it{s}} = 13.6 TeV, |#it{y}| < 1')
ratio_pinfo.Draw()

leg_ratio_models = ROOT.TLegend(0.58, 0.68, 0.92, 0.87)
leg_ratio_models.SetFillStyle(0)
leg_ratio_models.SetBorderSize(0)
leg_ratio_models.SetTextFont(42)
leg_ratio_models.SetMargin(0.1)
leg_ratio_models.SetTextSize(0.037)
leg_ratio_models.SetHeader('Total unc. = data #oplus model')
leg_ratio_models.AddEntry(ratio_congleton_band, 'Congleton', 'f')
leg_ratio_models.AddEntry(ratio_hwh_band, 'Congleton H.H. tune', 'f')
leg_ratio_models.AddEntry(ratio_gaus_band, 'Double #minus Gaussian', 'f')
leg_ratio_models.Draw()

c_ratio.SaveAs('ratio_data_tomcca.pdf')
c_ratio.SaveAs('ratio_data_tomcca.png')



output = ROOT.TFile('spectra_inel.root', 'RECREATE')
output.cd()
spectrum_stat.Write()
spectrum_syst.Write()
if levy:
    levy.Write('levy')
c.Write()
c_ratio.Write()
output.Close()


## get integrated yield from tomcca: multiply by x errors and sum
n_ev = 0
n_ev_gaus = 0
n_ev_hwh = 0
for i in range(gr.GetN()):
    n_ev += gr.GetY()[i] * (gr.GetErrorXlow(i) + gr.GetErrorXhigh(i))
    n_ev_gaus += gr_gaus.GetY()[i] * (gr_gaus.GetErrorXlow(i) + gr_gaus.GetErrorXhigh(i))
    n_ev_hwh += gr_hwh.GetY()[i] * (gr_hwh.GetErrorXlow(i) + gr_hwh.GetErrorXhigh(i))
print("TomCCA integrated yield:", n_ev)
print("TomCCA (gaus) integrated yield:", n_ev_gaus)
print("TomCCA (HWH) integrated yield:", n_ev_hwh)

## print N points for congleton and gaussian
print("Congleton points:", gr.GetN())
print("Gaussian points:", gr_gaus.GetN())
print("Congleton HWH points:", gr_hwh.GetN())
