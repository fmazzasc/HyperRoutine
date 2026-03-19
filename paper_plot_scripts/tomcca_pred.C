#ifdef __CLING__
#pragma cling optimize(0)
#endif
void tomcca_pred()
{
//=========Macro generated from canvas: c/c
//=========  (Tue Dec 16 10:26:17 2025) by ROOT version 6.36.04

   gStyle->SetCanvasPreferGL(kTRUE);

   TCanvas *c = new TCanvas("c", "c", 0, 0, 800, 600);
   gStyle->SetOptFit(1);
   gStyle->SetOptStat(1);
   gStyle->SetOptTitle(1);
   c->SetHighLightColor(2);
   TColor::SetPalette(57, nullptr);
   c->Range(-0.8875,-10.61324,6.3625,-7.351618);
   c->SetFillColor(0);
   c->SetBorderMode(0);
   c->SetBorderSize(2);
   c->SetLogy();
   c->SetTickx(1);
   c->SetTicky(1);
   c->SetLeftMargin(0.15);
   c->SetRightMargin(0.05);
   c->SetBottomMargin(0.12);
   c->SetFrameBorderMode(0);
   c->SetFrameBorderMode(0);
   
   TH1F *hframe__1 = new TH1F("hframe__1", "", 1000, 0.2, 6);
   hframe__1->SetMinimum(6e-11);
   hframe__1->SetMaximum(2.1e-08);
   hframe__1->SetDirectory(nullptr);
   hframe__1->SetStats(0);
   hframe__1->SetLineColor(TColor::GetColor("#000099"));
   hframe__1->GetXaxis()->SetTitle("#it{p}_{T} (GeV/#it{c})");
   hframe__1->GetXaxis()->SetLabelFont(42);
   hframe__1->GetXaxis()->SetLabelSize(0.03999999910593033);
   hframe__1->GetXaxis()->SetTitleSize(0.05000000074505806);
   hframe__1->GetXaxis()->SetTitleOffset(1.049999952316284);
   hframe__1->GetXaxis()->SetTitleFont(42);
   hframe__1->GetYaxis()->SetTitle("#frac{1}{#it{N}_{evt}}#frac{d^{2}#it{N}}{d#it{y}d#it{p}_{T}} (GeV/#it{c})^{-1}");
   hframe__1->GetYaxis()->SetLabelFont(42);
   hframe__1->GetYaxis()->SetLabelSize(0.03999999910593033);
   hframe__1->GetYaxis()->SetTitleSize(0.05000000074505806);
   hframe__1->GetYaxis()->SetTitleOffset(1.299999952316284);
   hframe__1->GetYaxis()->SetTitleFont(42);
   hframe__1->GetZaxis()->SetLabelFont(42);
   hframe__1->GetZaxis()->SetLabelSize(0.03999999910593033);
   hframe__1->GetZaxis()->SetTitleSize(0.05000000074505806);
   hframe__1->GetZaxis()->SetTitleOffset(1);
   hframe__1->GetZaxis()->SetTitleFont(42);
   hframe__1->Draw(" ");
   
   std::vector<Double_t> gre_fx_vect0{
      0.1, 0.3, 0.5, 0.7000000000000001, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9,
      2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9,
      4.1, 4.3, 4.5, 4.7, 4.9, 5.1, 5.3, 5.5, 5.7, 5.9,
      6.1, 6.3, 6.5, 6.7, 6.9, 7.1, 7.3, 7.5, 7.7, 7.9,
      8.1, 8.300000000000001, 8.5, 8.699999999999999, 8.9, 9.1, 9.300000000000001, 9.5, 9.700000000000001, 9.9
   };
   std::vector<Double_t> gre_fy_vect1{
      2.7925e-09, 9.584999999999999e-09, 1.373e-08, 1.699e-08, 1.75025e-08, 1.682e-08, 1.77725e-08, 1.669e-08, 1.56775e-08, 1.42125e-08,
      1.25475e-08, 1.1695e-08, 9.075e-09, 8.879999999999999e-09, 7.44e-09, 5.907499999999999e-09, 4.81e-09, 4.4175e-09, 2.885e-09, 2.8675e-09,
      2.2725e-09, 1.7325e-09, 1.38e-09, 9.7e-10, 8.649999999999999e-10, 7.624999999999999e-10, 6.449999999999999e-10, 4.175e-10, 3.55e-10, 4.075e-10,
      3.325e-10, 2.8e-10, 1.925e-10, 1.1e-10, 5e-11, 1.45e-10, 5.75e-11, 4.499999999999999e-11, 1.5e-11, 3.25e-11,
      1.75e-11, 3.25e-11, 2.75e-11, 2.5e-12, 5e-12, 5e-12, 2.5e-12, 5e-12, 7.5e-12, 1.25e-11
   };
   std::vector<Double_t> gre_fex_vect2{
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0
   };
   std::vector<Double_t> gre_fey_vect3{
      8.550000000000005e-10, 2.379999999999999e-09, 3.395000000000002e-09, 4.489999999999999e-09, 4.252499999999995e-09, 4.5375e-09, 4.244999999999996e-09, 4.700000000000002e-09, 4.054999999999999e-09, 3.850000000000001e-09,
      3.542499999999997e-09, 3.004999999999999e-09, 2.884999999999999e-09, 1.975000000000001e-09, 2.034999999999999e-09, 1.37e-09, 1.117499999999999e-09, 1.23e-09, 7.35e-10, 8.874999999999997e-10,
      4.999999999999999e-10, 2.75e-10, 3.175000000000002e-10, 4.024999999999998e-10, 2.1e-10, 2e-10, 1.475e-10, 1.475e-10, 4.000000000000002e-11, 4.999999999999998e-11,
      1.275e-10, 2.000000000000003e-11, 4.750000000000002e-11, 7.5e-12, 4.5e-11, 1.250000000000001e-11, 2.5e-11, 2.500000000000004e-12, 4.999999999999999e-12, 1.5e-11,
      1.5e-11, 1e-11, 2.5e-11, 0, 7.5e-12, 2.5e-12, 7.5e-12, 2.5e-12, 4.999999999999999e-12, 4.999999999999999e-12
   };
   TGraphErrors *gre = new TGraphErrors(50, gre_fx_vect0.data(), gre_fy_vect1.data(), gre_fex_vect2.data(), gre_fey_vect3.data());
   gre->SetName("Graph_from_ptDeuteronArgHistFineGaussCongle85");
   gre->SetTitle("ptDeuteronArgHistFineGaussCongle85");
   gre->SetFillColor(TColor::GetColor("#ff00007f"));
   gre->SetLineColor(TColor::GetColor("#ff0000"));
   
   TH1F *Graph_histogram1 = new TH1F("Graph_histogram1", "ptDeuteronArgHistFineGaussCongle85", 100, 0, 10.88);
   Graph_histogram1->SetMinimum(7.499999999999999e-13);
   Graph_histogram1->SetMaximum(2.421916666666666e-08);
   Graph_histogram1->SetDirectory(nullptr);
   Graph_histogram1->SetStats(0);
   Graph_histogram1->SetLineColor(TColor::GetColor("#000099"));
   Graph_histogram1->GetXaxis()->SetLabelFont(42);
   Graph_histogram1->GetXaxis()->SetLabelSize(0.03999999910593033);
   Graph_histogram1->GetXaxis()->SetTitleSize(0.05000000074505806);
   Graph_histogram1->GetXaxis()->SetTitleOffset(1.049999952316284);
   Graph_histogram1->GetXaxis()->SetTitleFont(42);
   Graph_histogram1->GetYaxis()->SetLabelFont(42);
   Graph_histogram1->GetYaxis()->SetLabelSize(0.03999999910593033);
   Graph_histogram1->GetYaxis()->SetTitleSize(0.05000000074505806);
   Graph_histogram1->GetYaxis()->SetTitleOffset(1.100000023841858);
   Graph_histogram1->GetYaxis()->SetTitleFont(42);
   Graph_histogram1->GetZaxis()->SetLabelFont(42);
   Graph_histogram1->GetZaxis()->SetLabelSize(0.03999999910593033);
   Graph_histogram1->GetZaxis()->SetTitleSize(0.05000000074505806);
   Graph_histogram1->GetZaxis()->SetTitleOffset(1);
   Graph_histogram1->GetZaxis()->SetTitleFont(42);
   gre->SetHistogram(Graph_histogram1);
   
   gre->Draw("3 ");
   
   std::vector<Double_t> gre_fx_vect4{
      0.1, 0.3, 0.5, 0.7000000000000001, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9,
      2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9,
      4.1, 4.3, 4.5, 4.7, 4.9, 5.1, 5.3, 5.5, 5.7, 5.9,
      6.1, 6.3, 6.5, 6.7, 6.9, 7.1, 7.3, 7.5, 7.7, 7.9,
      8.1, 8.300000000000001, 8.5, 8.699999999999999, 8.9, 9.1, 9.300000000000001, 9.5, 9.700000000000001, 9.9
   };
   std::vector<Double_t> gre_fy_vect5{
      4.6e-10, 4.085e-09, 7.137499999999999e-09, 8.5625e-09, 8.249999999999999e-09, 7.534999999999999e-09, 8.594999999999999e-09, 7.9425e-09, 6.274999999999999e-09, 5.295e-09,
      4.7775e-09, 4.54e-09, 3.085e-09, 2.557499999999999e-09, 1.975e-09, 1.4175e-09, 1.2075e-09, 1.2825e-09, 6.324999999999999e-10, 7.624999999999999e-10,
      4.625e-10, 2.4e-10, 1.775e-10, 1.45e-10, 1.425e-10, 8.999999999999999e-11, 9.5e-11, 7.749999999999999e-11, 5.499999999999999e-11, 4.75e-11,
      2e-11, 1.25e-11, 2.25e-11, 1.75e-11, 5e-12, 7.5e-12, 2.5e-12, 9.999999999999999e-12, 0, 0,
      0, 0, 2.5e-12, 0, 0, 0, 0, 0, 0, 0
   };
   std::vector<Double_t> gre_fex_vect6{
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0
   };
   std::vector<Double_t> gre_fey_vect7{
      1.425e-10, 8.949999999999999e-10, 1.447499999999999e-09, 2.092499999999999e-09, 2.14e-09, 1.8e-09, 2.005000000000002e-09, 1.627499999999999e-09, 1.527500000000001e-09, 1.355e-09,
      1.254999999999999e-09, 8.325000000000002e-10, 7.674999999999997e-10, 5.275000000000001e-10, 5.250000000000002e-10, 5.674999999999997e-10, 2.050000000000002e-10, 2.200000000000001e-10, 2.8e-10, 1.15e-10,
      1.625e-10, 5.750000000000001e-11, 5.749999999999998e-11, 7.249999999999998e-11, 7.5e-11, 1e-11, 5.250000000000001e-11, 2.499999999999991e-12, 2.25e-11, 1.75e-11,
      7.499999999999997e-12, 9.999999999999998e-12, 2.499999999999998e-12, 2.500000000000001e-12, 0, 2.5e-12, 5.000000000000001e-12, 2.499999999999999e-12, 0, 0,
      0, 2.5e-12, 0, 0, 0, 0, 0, 0, 0, 0
   };
   gre = new TGraphErrors(50, gre_fx_vect4.data(), gre_fy_vect5.data(), gre_fex_vect6.data(), gre_fey_vect7.data());
   gre->SetName("Graph_from_ptDeuteronArgHistFineAV85");
   gre->SetTitle("ptDeuteronArgHistFineAV85");
   gre->SetFillColor(TColor::GetColor("#ffcc007f"));
   gre->SetLineColor(TColor::GetColor("#ffcc00"));
   
   TH1F *Graph_histogram2 = new TH1F("Graph_histogram2", "ptDeuteronArgHistFineAV85", 100, 0, 10.88);
   Graph_histogram2->SetMinimum(1.17205e-11);
   Graph_histogram2->SetMaximum(1.17205e-08);
   Graph_histogram2->SetDirectory(nullptr);
   Graph_histogram2->SetStats(0);
   Graph_histogram2->SetLineColor(TColor::GetColor("#000099"));
   Graph_histogram2->GetXaxis()->SetLabelFont(42);
   Graph_histogram2->GetXaxis()->SetLabelSize(0.03999999910593033);
   Graph_histogram2->GetXaxis()->SetTitleSize(0.05000000074505806);
   Graph_histogram2->GetXaxis()->SetTitleOffset(1.049999952316284);
   Graph_histogram2->GetXaxis()->SetTitleFont(42);
   Graph_histogram2->GetYaxis()->SetLabelFont(42);
   Graph_histogram2->GetYaxis()->SetLabelSize(0.03999999910593033);
   Graph_histogram2->GetYaxis()->SetTitleSize(0.05000000074505806);
   Graph_histogram2->GetYaxis()->SetTitleOffset(1.100000023841858);
   Graph_histogram2->GetYaxis()->SetTitleFont(42);
   Graph_histogram2->GetZaxis()->SetLabelFont(42);
   Graph_histogram2->GetZaxis()->SetLabelSize(0.03999999910593033);
   Graph_histogram2->GetZaxis()->SetTitleSize(0.05000000074505806);
   Graph_histogram2->GetZaxis()->SetTitleOffset(1);
   Graph_histogram2->GetZaxis()->SetTitleFont(42);
   gre->SetHistogram(Graph_histogram2);
   
   gre->Draw("3 ");
   
   std::vector<Double_t> h_default_spectrum_stat__2_x_vect8{ 1.4, 1.7, 2, 2.3, 2.6, 3, 3.5, 4, 5 };
   TH1D *h_default_spectrum_stat__2 = new TH1D("h_default_spectrum_stat__2", "", 8, h_default_spectrum_stat__2_x_vect8.data());
   h_default_spectrum_stat__2->SetBinContent(1,8.250789620671916e-09);
   h_default_spectrum_stat__2->SetBinContent(2,5.962328635556963e-09);
   h_default_spectrum_stat__2->SetBinContent(3,3.544846163423816e-09);
   h_default_spectrum_stat__2->SetBinContent(4,2.696848705840286e-09);
   h_default_spectrum_stat__2->SetBinContent(5,1.948387031747601e-09);
   h_default_spectrum_stat__2->SetBinContent(6,1.162503478400591e-09);
   h_default_spectrum_stat__2->SetBinContent(7,5.438739690298099e-10);
   h_default_spectrum_stat__2->SetBinContent(8,2.1967446707029e-10);
   h_default_spectrum_stat__2->SetBinError(1,1.335605834419267e-09);
   h_default_spectrum_stat__2->SetBinError(2,6.002967121421738e-10);
   h_default_spectrum_stat__2->SetBinError(3,3.869829587782087e-10);
   h_default_spectrum_stat__2->SetBinError(4,3.204482827155134e-10);
   h_default_spectrum_stat__2->SetBinError(5,2.071638120116265e-10);
   h_default_spectrum_stat__2->SetBinError(6,1.360707539534957e-10);
   h_default_spectrum_stat__2->SetBinError(7,8.543435544025619e-11);
   h_default_spectrum_stat__2->SetBinError(8,4.089876249335419e-11);
   h_default_spectrum_stat__2->SetEntries(8);
   h_default_spectrum_stat__2->SetDirectory(nullptr);
   h_default_spectrum_stat__2->SetStats(0);
   h_default_spectrum_stat__2->SetFillStyle(0);
   h_default_spectrum_stat__2->SetLineColor(TColor::GetColor("#0066cc"));
   h_default_spectrum_stat__2->SetMarkerColor(TColor::GetColor("#0066cc"));
   h_default_spectrum_stat__2->SetMarkerStyle(20);
   h_default_spectrum_stat__2->GetXaxis()->SetTitle("p_{T} (GeV/c)");
   h_default_spectrum_stat__2->GetXaxis()->SetLabelFont(42);
   h_default_spectrum_stat__2->GetXaxis()->SetLabelSize(0.03999999910593033);
   h_default_spectrum_stat__2->GetXaxis()->SetTitleSize(0.05000000074505806);
   h_default_spectrum_stat__2->GetXaxis()->SetTitleOffset(1.049999952316284);
   h_default_spectrum_stat__2->GetXaxis()->SetTitleFont(42);
   h_default_spectrum_stat__2->GetYaxis()->SetTitle("Counts");
   h_default_spectrum_stat__2->GetYaxis()->SetLabelFont(42);
   h_default_spectrum_stat__2->GetYaxis()->SetLabelSize(0.03999999910593033);
   h_default_spectrum_stat__2->GetYaxis()->SetTitleSize(0.05000000074505806);
   h_default_spectrum_stat__2->GetYaxis()->SetTitleOffset(1.100000023841858);
   h_default_spectrum_stat__2->GetYaxis()->SetTitleFont(42);
   h_default_spectrum_stat__2->GetZaxis()->SetLabelFont(42);
   h_default_spectrum_stat__2->GetZaxis()->SetLabelSize(0.03999999910593033);
   h_default_spectrum_stat__2->GetZaxis()->SetTitleSize(0.05000000074505806);
   h_default_spectrum_stat__2->GetZaxis()->SetTitleOffset(1);
   h_default_spectrum_stat__2->GetZaxis()->SetTitleFont(42);
   h_default_spectrum_stat__2->Draw("PEX0 SAME");
   
   std::vector<Double_t> h_default_spectrum_syst__3_x_vect9{ 1.4, 1.7, 2, 2.3, 2.6, 3, 3.5, 4, 5 };
   TH1D *h_default_spectrum_syst__3 = new TH1D("h_default_spectrum_syst__3", "", 8, h_default_spectrum_syst__3_x_vect9.data());
   h_default_spectrum_syst__3->SetBinContent(1,8.250789620671916e-09);
   h_default_spectrum_syst__3->SetBinContent(2,5.962328635556963e-09);
   h_default_spectrum_syst__3->SetBinContent(3,3.544846163423816e-09);
   h_default_spectrum_syst__3->SetBinContent(4,2.696848705840286e-09);
   h_default_spectrum_syst__3->SetBinContent(5,1.948387031747601e-09);
   h_default_spectrum_syst__3->SetBinContent(6,1.162503478400591e-09);
   h_default_spectrum_syst__3->SetBinContent(7,5.438739690298099e-10);
   h_default_spectrum_syst__3->SetBinContent(8,2.1967446707029e-10);
   h_default_spectrum_syst__3->SetBinError(1,1.050012058345458e-09);
   h_default_spectrum_syst__3->SetBinError(2,6.524612886804203e-10);
   h_default_spectrum_syst__3->SetBinError(3,3.770842834804962e-10);
   h_default_spectrum_syst__3->SetBinError(4,3.048292735263033e-10);
   h_default_spectrum_syst__3->SetBinError(5,2.165225189553991e-10);
   h_default_spectrum_syst__3->SetBinError(6,1.24956569252068e-10);
   h_default_spectrum_syst__3->SetBinError(7,5.930809215717393e-11);
   h_default_spectrum_syst__3->SetBinError(8,2.578900776875817e-11);
   h_default_spectrum_syst__3->SetEntries(8);
   h_default_spectrum_syst__3->SetDirectory(nullptr);
   h_default_spectrum_syst__3->SetStats(0);
   h_default_spectrum_syst__3->SetFillStyle(0);
   h_default_spectrum_syst__3->SetLineColor(TColor::GetColor("#0066cc"));
   h_default_spectrum_syst__3->SetMarkerColor(TColor::GetColor("#0066cc"));
   h_default_spectrum_syst__3->SetMarkerStyle(20);
   h_default_spectrum_syst__3->GetXaxis()->SetTitle("p_{T} (GeV/c)");
   h_default_spectrum_syst__3->GetXaxis()->SetLabelFont(42);
   h_default_spectrum_syst__3->GetXaxis()->SetLabelSize(0.03999999910593033);
   h_default_spectrum_syst__3->GetXaxis()->SetTitleSize(0.05000000074505806);
   h_default_spectrum_syst__3->GetXaxis()->SetTitleOffset(1.049999952316284);
   h_default_spectrum_syst__3->GetXaxis()->SetTitleFont(42);
   h_default_spectrum_syst__3->GetYaxis()->SetTitle("Counts");
   h_default_spectrum_syst__3->GetYaxis()->SetLabelFont(42);
   h_default_spectrum_syst__3->GetYaxis()->SetLabelSize(0.03999999910593033);
   h_default_spectrum_syst__3->GetYaxis()->SetTitleSize(0.05000000074505806);
   h_default_spectrum_syst__3->GetYaxis()->SetTitleOffset(1.100000023841858);
   h_default_spectrum_syst__3->GetYaxis()->SetTitleFont(42);
   h_default_spectrum_syst__3->GetZaxis()->SetLabelFont(42);
   h_default_spectrum_syst__3->GetZaxis()->SetLabelSize(0.03999999910593033);
   h_default_spectrum_syst__3->GetZaxis()->SetTitleSize(0.05000000074505806);
   h_default_spectrum_syst__3->GetZaxis()->SetTitleOffset(1);
   h_default_spectrum_syst__3->GetZaxis()->SetTitleFont(42);
   h_default_spectrum_syst__3->Draw("PE2 SAME");
   
   TLegend *leg = new TLegend(0.19, 0.36, 0.6, 0.46, nullptr, "brNDC");
   leg->SetBorderSize(0);
   leg->SetTextSize(0.037);
   leg->SetLineColor(1);
   leg->SetLineStyle(1);
   leg->SetLineWidth(1);
   leg->SetFillColor(0);
   leg->SetFillStyle(0);
   TLegendEntry *legentry = leg->AddEntry("h_default_spectrum_stat","{}^{3}_{#bar{#Lambda}}#bar{H}#kern[0.3]{#rightarrow}#kern[0.1]{^{3}#bar{He}}#kern[0.3]{+}#kern[0.3]{#pi^{+}}, |#it{y}| < 1","PE");
   legentry->SetLineColor(TColor::GetColor("#0066cc"));
   legentry->SetMarkerColor(TColor::GetColor("#0066cc"));
   legentry->SetMarkerStyle(20);
   legentry->SetTextFont(42);
   leg->Draw();
   
   leg = new TLegend(0.19, 0.15, 0.65, 0.36, nullptr, "brNDC");
   leg->SetBorderSize(0);
   leg->SetTextSize(0.037);
   leg->SetLineColor(1);
   leg->SetLineStyle(1);
   leg->SetLineWidth(1);
   leg->SetFillColor(0);
   leg->SetFillStyle(0);
   legentry = leg->AddEntry("NULL","\tarXiv:2504.02491 (2025)","h");
   legentry->SetTextFont(42);
   legentry = leg->AddEntry("Graph_from_ptDeuteronArgHistFineGaussCongle85","Congleton + AV","F");
   legentry->SetFillColor(TColor::GetColor("#ff00007f"));
   legentry->SetFillStyle(1001);
   legentry->SetLineColor(TColor::GetColor("#ff0000"));
   legentry->SetTextFont(42);
   legentry = leg->AddEntry("Graph_from_ptDeuteronArgHistFineAV85","Congleton + Gaus (HWH)","F");
   legentry->SetFillColor(TColor::GetColor("#ffcc007f"));
   legentry->SetFillStyle(1001);
   legentry->SetLineColor(TColor::GetColor("#ffcc00"));
   legentry->SetTextFont(42);
   leg->Draw();
   
   TPaveText *pt = new TPaveText(0.53, 0.6, 0.89, 0.85, "brNDC");
   pt->SetBorderSize(0);
   pt->SetFillStyle(0);
   pt->SetTextAlign(11);
   pt->SetTextFont(42);
   TText *pt_text0 = pt->AddText("ALICE");
   TText *pt_text1 = pt->AddText("Run 3 pp,#kern[0.4]{#sqrt{#it{s}}} = 13.6 TeV");
   TText *pt_text2 = pt->AddText("N_{evt} = 1.9 #times 10^{12}");
   TText *pt_text3 = pt->AddText("#pm 10% global unc. not shown");
   pt->Draw("brNDC");
   c->Modified();
   c->SetSelected(c);
}
