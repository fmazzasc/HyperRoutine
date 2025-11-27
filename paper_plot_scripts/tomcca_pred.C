#ifdef __CLING__
#pragma cling optimize(0)
#endif
void tomcca_pred()
{
//=========Macro generated from canvas: c/c
//=========  (Tue Nov 25 14:49:37 2025) by ROOT version 6.32.06

   gStyle->SetCanvasPreferGL(kTRUE);

   TCanvas *c = new TCanvas("c", "c",0,0,800,600);
   gStyle->SetOptFit(1);
   c->SetHighLightColor(2);
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
   
   TH1F *hframe__1 = new TH1F("hframe__1","",1000,0.2,6);
   hframe__1->SetMinimum(6e-11);
   hframe__1->SetMaximum(2.1e-08);
   hframe__1->SetDirectory(nullptr);
   hframe__1->SetStats(0);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#000099");
   hframe__1->SetLineColor(ci);
   hframe__1->GetXaxis()->SetTitle("#it{p}_{T} (GeV/#it{c})");
   hframe__1->GetXaxis()->SetLabelFont(42);
   hframe__1->GetXaxis()->SetLabelSize(0.04);
   hframe__1->GetXaxis()->SetTitleSize(0.05);
   hframe__1->GetXaxis()->SetTitleOffset(1.05);
   hframe__1->GetXaxis()->SetTitleFont(42);
   hframe__1->GetYaxis()->SetTitle("#frac{1}{#it{N}_{evt}}#frac{d^{2}#it{N}}{d#it{y}d#it{p}_{T}} (GeV/#it{c})^{-1}");
   hframe__1->GetYaxis()->SetLabelFont(42);
   hframe__1->GetYaxis()->SetLabelSize(0.04);
   hframe__1->GetYaxis()->SetTitleSize(0.05);
   hframe__1->GetYaxis()->SetTitleOffset(1.3);
   hframe__1->GetYaxis()->SetTitleFont(42);
   hframe__1->GetZaxis()->SetLabelFont(42);
   hframe__1->GetZaxis()->SetLabelSize(0.04);
   hframe__1->GetZaxis()->SetTitleSize(0.05);
   hframe__1->GetZaxis()->SetTitleOffset(1);
   hframe__1->GetZaxis()->SetTitleFont(42);
   hframe__1->Draw(" ");
   
   Double_t Graph_from_ptDeuteronArgFine_fx1001[25] = { 0.2, 0.6, 1, 1.4, 1.8, 2.2, 2.6, 3, 3.4, 3.8, 4.2, 4.6, 5, 5.4, 5.8, 6.2, 6.6,
   7, 7.4, 7.8, 8.2, 8.6, 9, 9.4, 9.8 };
   Double_t Graph_from_ptDeuteronArgFine_fy1001[25] = { 6.944444e-09, 1.087674e-08, 1.172917e-08, 9.618056e-09, 6.590278e-09, 3.949653e-09, 2.454861e-09, 1.260417e-09, 6.5625e-10, 3.142361e-10, 1.527778e-10, 9.895833e-11, 6.25e-11, 1.041667e-11, 8.680556e-12, 3.472222e-12, 6.944444e-12,
   0, 0, 3.472222e-12, 0, 0, 0, 0, 0 };
   Double_t Graph_from_ptDeuteronArgFine_fex1001[25] = { 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 };
   Double_t Graph_from_ptDeuteronArgFine_fey1001[25] = { 1.098013e-10, 1.374162e-10, 1.426995e-10, 1.292208e-10, 1.069647e-10, 8.280722e-11, 6.528332e-11, 4.677845e-11, 3.375386e-11, 2.335699e-11, 1.628617e-11, 1.310735e-11, 1.041667e-11, 4.252586e-12, 3.882062e-12, 2.455232e-12, 3.472222e-12,
   0, 0, 2.455232e-12, 0, 0, 0, 0, 0 };
   TGraphErrors *gre = new TGraphErrors(25,Graph_from_ptDeuteronArgFine_fx1001,Graph_from_ptDeuteronArgFine_fy1001,Graph_from_ptDeuteronArgFine_fex1001,Graph_from_ptDeuteronArgFine_fey1001);
   gre->SetName("Graph_from_ptDeuteronArgFine");
   gre->SetTitle("ptDeuteronArgFine");

   ci = 1204;
   color = new TColor(ci, 1, 0, 0, " ", 0.5);
   gre->SetFillColor(ci);

   ci = TColor::GetColor("#ff0000");
   gre->SetLineColor(ci);
   gre->SetLineWidth(2);

   ci = TColor::GetColor("#ff0000");
   gre->SetMarkerColor(ci);
   gre->SetMarkerSize(0);
   
   TH1F *Graph_Graph_from_ptDeuteronArgFine1001 = new TH1F("Graph_Graph_from_ptDeuteronArgFine1001","ptDeuteronArgFine",100,0,11);
   Graph_Graph_from_ptDeuteronArgFine1001->SetMinimum(1.305905e-11);
   Graph_Graph_from_ptDeuteronArgFine1001->SetMaximum(1.305905e-08);
   Graph_Graph_from_ptDeuteronArgFine1001->SetDirectory(nullptr);
   Graph_Graph_from_ptDeuteronArgFine1001->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph_from_ptDeuteronArgFine1001->SetLineColor(ci);
   Graph_Graph_from_ptDeuteronArgFine1001->GetXaxis()->SetLabelFont(42);
   Graph_Graph_from_ptDeuteronArgFine1001->GetXaxis()->SetLabelSize(0.04);
   Graph_Graph_from_ptDeuteronArgFine1001->GetXaxis()->SetTitleSize(0.05);
   Graph_Graph_from_ptDeuteronArgFine1001->GetXaxis()->SetTitleOffset(1.05);
   Graph_Graph_from_ptDeuteronArgFine1001->GetXaxis()->SetTitleFont(42);
   Graph_Graph_from_ptDeuteronArgFine1001->GetYaxis()->SetLabelFont(42);
   Graph_Graph_from_ptDeuteronArgFine1001->GetYaxis()->SetLabelSize(0.04);
   Graph_Graph_from_ptDeuteronArgFine1001->GetYaxis()->SetTitleSize(0.05);
   Graph_Graph_from_ptDeuteronArgFine1001->GetYaxis()->SetTitleOffset(1.1);
   Graph_Graph_from_ptDeuteronArgFine1001->GetYaxis()->SetTitleFont(42);
   Graph_Graph_from_ptDeuteronArgFine1001->GetZaxis()->SetLabelFont(42);
   Graph_Graph_from_ptDeuteronArgFine1001->GetZaxis()->SetLabelSize(0.04);
   Graph_Graph_from_ptDeuteronArgFine1001->GetZaxis()->SetTitleSize(0.05);
   Graph_Graph_from_ptDeuteronArgFine1001->GetZaxis()->SetTitleOffset(1);
   Graph_Graph_from_ptDeuteronArgFine1001->GetZaxis()->SetTitleFont(42);
   gre->SetHistogram(Graph_Graph_from_ptDeuteronArgFine1001);
   
   gre->Draw("3 ");
   
   Double_t Graph_from_ptDeuteronArgFine_fx1002[25] = { 0.2, 0.6, 1, 1.4, 1.8, 2.2, 2.6, 3, 3.4, 3.8, 4.2, 4.6, 5, 5.4, 5.8, 6.2, 6.6,
   7, 7.4, 7.8, 8.2, 8.6, 9, 9.4, 9.8 };
   Double_t Graph_from_ptDeuteronArgFine_fy1002[25] = { 3.715278e-09, 5.743056e-09, 6.703125e-09, 5.138889e-09, 3.621528e-09, 1.9375e-09, 1.25e-09, 5.381944e-10, 3.368056e-10, 9.722222e-11, 5.902778e-11, 3.472222e-11, 3.472222e-12, 5.208333e-12, 5.208333e-12, 5.208333e-12, 3.472222e-12,
   0, 1.736111e-12, 0, 0, 0, 0, 0, 0 };
   Double_t Graph_from_ptDeuteronArgFine_fex1002[25] = { 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 };
   Double_t Graph_from_ptDeuteronArgFine_fey1002[25] = { 8.031273e-11, 9.98528e-11, 1.078766e-10, 9.445466e-11, 7.929297e-11, 5.799755e-11, 4.658475e-11, 3.056739e-11, 2.418123e-11, 1.299187e-11, 1.012318e-11, 7.764125e-12, 2.455232e-12, 3.007033e-12, 3.007033e-12, 3.007033e-12, 2.455232e-12,
   0, 1.736111e-12, 0, 0, 0, 0, 0, 0 };
   gre = new TGraphErrors(25,Graph_from_ptDeuteronArgFine_fx1002,Graph_from_ptDeuteronArgFine_fy1002,Graph_from_ptDeuteronArgFine_fex1002,Graph_from_ptDeuteronArgFine_fey1002);
   gre->SetName("Graph_from_ptDeuteronArgFine");
   gre->SetTitle("ptDeuteronArgFine");

   ci = 1205;
   color = new TColor(ci, 1, 0.8, 0, " ", 0.5);
   gre->SetFillColor(ci);

   ci = TColor::GetColor("#ffcc00");
   gre->SetLineColor(ci);
   gre->SetLineWidth(2);

   ci = TColor::GetColor("#ffcc00");
   gre->SetMarkerColor(ci);
   gre->SetMarkerSize(0);
   
   TH1F *Graph_Graph_from_ptDeuteronArgFine1002 = new TH1F("Graph_Graph_from_ptDeuteronArgFine1002","ptDeuteronArgFine",100,0,11);
   Graph_Graph_from_ptDeuteronArgFine1002->SetMinimum(7.492102e-12);
   Graph_Graph_from_ptDeuteronArgFine1002->SetMaximum(7.492102e-09);
   Graph_Graph_from_ptDeuteronArgFine1002->SetDirectory(nullptr);
   Graph_Graph_from_ptDeuteronArgFine1002->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph_from_ptDeuteronArgFine1002->SetLineColor(ci);
   Graph_Graph_from_ptDeuteronArgFine1002->GetXaxis()->SetLabelFont(42);
   Graph_Graph_from_ptDeuteronArgFine1002->GetXaxis()->SetLabelSize(0.04);
   Graph_Graph_from_ptDeuteronArgFine1002->GetXaxis()->SetTitleSize(0.05);
   Graph_Graph_from_ptDeuteronArgFine1002->GetXaxis()->SetTitleOffset(1.05);
   Graph_Graph_from_ptDeuteronArgFine1002->GetXaxis()->SetTitleFont(42);
   Graph_Graph_from_ptDeuteronArgFine1002->GetYaxis()->SetLabelFont(42);
   Graph_Graph_from_ptDeuteronArgFine1002->GetYaxis()->SetLabelSize(0.04);
   Graph_Graph_from_ptDeuteronArgFine1002->GetYaxis()->SetTitleSize(0.05);
   Graph_Graph_from_ptDeuteronArgFine1002->GetYaxis()->SetTitleOffset(1.1);
   Graph_Graph_from_ptDeuteronArgFine1002->GetYaxis()->SetTitleFont(42);
   Graph_Graph_from_ptDeuteronArgFine1002->GetZaxis()->SetLabelFont(42);
   Graph_Graph_from_ptDeuteronArgFine1002->GetZaxis()->SetLabelSize(0.04);
   Graph_Graph_from_ptDeuteronArgFine1002->GetZaxis()->SetTitleSize(0.05);
   Graph_Graph_from_ptDeuteronArgFine1002->GetZaxis()->SetTitleOffset(1);
   Graph_Graph_from_ptDeuteronArgFine1002->GetZaxis()->SetTitleFont(42);
   gre->SetHistogram(Graph_Graph_from_ptDeuteronArgFine1002);
   
   gre->Draw("3 ");
   Double_t xAxis1[9] = {1.4, 1.7, 2, 2.3, 2.6, 3, 3.5, 4, 5}; 
   
   TH1D *h_default_spectrum_stat__2 = new TH1D("h_default_spectrum_stat__2","",8, xAxis1);
   h_default_spectrum_stat__2->SetBinContent(1,8.247829e-09);
   h_default_spectrum_stat__2->SetBinContent(2,5.962616e-09);
   h_default_spectrum_stat__2->SetBinContent(3,3.545261e-09);
   h_default_spectrum_stat__2->SetBinContent(4,2.890827e-09);
   h_default_spectrum_stat__2->SetBinContent(5,1.948387e-09);
   h_default_spectrum_stat__2->SetBinContent(6,1.162564e-09);
   h_default_spectrum_stat__2->SetBinContent(7,5.44081e-10);
   h_default_spectrum_stat__2->SetBinContent(8,2.194767e-10);
   h_default_spectrum_stat__2->SetBinError(1,1.33547e-09);
   h_default_spectrum_stat__2->SetBinError(2,6.003175e-10);
   h_default_spectrum_stat__2->SetBinError(3,3.86995e-10);
   h_default_spectrum_stat__2->SetBinError(4,3.316834e-10);
   h_default_spectrum_stat__2->SetBinError(5,2.071638e-10);
   h_default_spectrum_stat__2->SetBinError(6,1.360894e-10);
   h_default_spectrum_stat__2->SetBinError(7,8.473881e-11);
   h_default_spectrum_stat__2->SetBinError(8,4.082354e-11);
   h_default_spectrum_stat__2->SetEntries(8);
   h_default_spectrum_stat__2->SetDirectory(nullptr);
   h_default_spectrum_stat__2->SetStats(0);
   h_default_spectrum_stat__2->SetFillStyle(0);

   ci = TColor::GetColor("#0066cc");
   h_default_spectrum_stat__2->SetLineColor(ci);

   ci = TColor::GetColor("#0066cc");
   h_default_spectrum_stat__2->SetMarkerColor(ci);
   h_default_spectrum_stat__2->SetMarkerStyle(20);
   h_default_spectrum_stat__2->GetXaxis()->SetTitle("p_{T} (GeV/c)");
   h_default_spectrum_stat__2->GetXaxis()->SetLabelFont(42);
   h_default_spectrum_stat__2->GetXaxis()->SetLabelSize(0.04);
   h_default_spectrum_stat__2->GetXaxis()->SetTitleSize(0.05);
   h_default_spectrum_stat__2->GetXaxis()->SetTitleOffset(1.05);
   h_default_spectrum_stat__2->GetXaxis()->SetTitleFont(42);
   h_default_spectrum_stat__2->GetYaxis()->SetTitle("Counts");
   h_default_spectrum_stat__2->GetYaxis()->SetLabelFont(42);
   h_default_spectrum_stat__2->GetYaxis()->SetLabelSize(0.04);
   h_default_spectrum_stat__2->GetYaxis()->SetTitleSize(0.05);
   h_default_spectrum_stat__2->GetYaxis()->SetTitleOffset(1.1);
   h_default_spectrum_stat__2->GetYaxis()->SetTitleFont(42);
   h_default_spectrum_stat__2->GetZaxis()->SetLabelFont(42);
   h_default_spectrum_stat__2->GetZaxis()->SetLabelSize(0.04);
   h_default_spectrum_stat__2->GetZaxis()->SetTitleSize(0.05);
   h_default_spectrum_stat__2->GetZaxis()->SetTitleOffset(1);
   h_default_spectrum_stat__2->GetZaxis()->SetTitleFont(42);
   h_default_spectrum_stat__2->Draw("PEX0 SAME");
   Double_t xAxis2[9] = {1.4, 1.7, 2, 2.3, 2.6, 3, 3.5, 4, 5}; 
   
   TH1D *h_default_spectrum_syst__3 = new TH1D("h_default_spectrum_syst__3","",8, xAxis2);
   h_default_spectrum_syst__3->SetBinContent(1,8.247829e-09);
   h_default_spectrum_syst__3->SetBinContent(2,5.962616e-09);
   h_default_spectrum_syst__3->SetBinContent(3,3.545261e-09);
   h_default_spectrum_syst__3->SetBinContent(4,2.890827e-09);
   h_default_spectrum_syst__3->SetBinContent(5,1.948387e-09);
   h_default_spectrum_syst__3->SetBinContent(6,1.162564e-09);
   h_default_spectrum_syst__3->SetBinContent(7,5.44081e-10);
   h_default_spectrum_syst__3->SetBinContent(8,2.194767e-10);
   h_default_spectrum_syst__3->SetBinError(1,1.044835e-09);
   h_default_spectrum_syst__3->SetBinError(2,6.525828e-10);
   h_default_spectrum_syst__3->SetBinError(3,3.772095e-10);
   h_default_spectrum_syst__3->SetBinError(4,3.237707e-10);
   h_default_spectrum_syst__3->SetBinError(5,2.124291e-10);
   h_default_spectrum_syst__3->SetBinError(6,1.245464e-10);
   h_default_spectrum_syst__3->SetBinError(7,5.898308e-11);
   h_default_spectrum_syst__3->SetBinError(8,2.567348e-11);
   h_default_spectrum_syst__3->SetEntries(8);
   h_default_spectrum_syst__3->SetDirectory(nullptr);
   h_default_spectrum_syst__3->SetStats(0);
   h_default_spectrum_syst__3->SetFillStyle(0);

   ci = TColor::GetColor("#0066cc");
   h_default_spectrum_syst__3->SetLineColor(ci);

   ci = TColor::GetColor("#0066cc");
   h_default_spectrum_syst__3->SetMarkerColor(ci);
   h_default_spectrum_syst__3->SetMarkerStyle(20);
   h_default_spectrum_syst__3->GetXaxis()->SetTitle("p_{T} (GeV/c)");
   h_default_spectrum_syst__3->GetXaxis()->SetLabelFont(42);
   h_default_spectrum_syst__3->GetXaxis()->SetLabelSize(0.04);
   h_default_spectrum_syst__3->GetXaxis()->SetTitleSize(0.05);
   h_default_spectrum_syst__3->GetXaxis()->SetTitleOffset(1.05);
   h_default_spectrum_syst__3->GetXaxis()->SetTitleFont(42);
   h_default_spectrum_syst__3->GetYaxis()->SetTitle("Counts");
   h_default_spectrum_syst__3->GetYaxis()->SetLabelFont(42);
   h_default_spectrum_syst__3->GetYaxis()->SetLabelSize(0.04);
   h_default_spectrum_syst__3->GetYaxis()->SetTitleSize(0.05);
   h_default_spectrum_syst__3->GetYaxis()->SetTitleOffset(1.1);
   h_default_spectrum_syst__3->GetYaxis()->SetTitleFont(42);
   h_default_spectrum_syst__3->GetZaxis()->SetLabelFont(42);
   h_default_spectrum_syst__3->GetZaxis()->SetLabelSize(0.04);
   h_default_spectrum_syst__3->GetZaxis()->SetTitleSize(0.05);
   h_default_spectrum_syst__3->GetZaxis()->SetTitleOffset(1);
   h_default_spectrum_syst__3->GetZaxis()->SetTitleFont(42);
   h_default_spectrum_syst__3->Draw("PE2 SAME");
   
   TLegend *leg = new TLegend(0.19,0.36,0.6,0.46,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetTextSize(0.037);
   leg->SetLineColor(1);
   leg->SetLineStyle(1);
   leg->SetLineWidth(1);
   leg->SetFillColor(0);
   leg->SetFillStyle(0);
   TLegendEntry *entry=leg->AddEntry("h_default_spectrum_stat","{}^{3}_{#bar{#Lambda}}#bar{H}#kern[0.3]{#rightarrow}#kern[0.1]{^{3}#bar{He}}#kern[0.3]{+}#kern[0.3]{#pi^{+}}, |#it{y}| < 1","PE");

   ci = TColor::GetColor("#0066cc");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#0066cc");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   leg->Draw();
   
   leg = new TLegend(0.19,0.15,0.65,0.36,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetTextSize(0.037);
   leg->SetLineColor(1);
   leg->SetLineStyle(1);
   leg->SetLineWidth(1);
   leg->SetFillColor(0);
   leg->SetFillStyle(0);
   entry=leg->AddEntry("NULL","	arXiv:2504.02491 (2025)","h");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("Graph_from_ptDeuteronArgFine","Congleton Wave Function","F");

   ci = 1204;
   color = new TColor(ci, 1, 0, 0, " ", 0.5);
   entry->SetFillColor(ci);
   entry->SetFillStyle(1001);

   ci = TColor::GetColor("#ff0000");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(2);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("Graph_from_ptDeuteronArgFine","d#minus#Lambda Gaussian Wave Function","F");

   ci = 1205;
   color = new TColor(ci, 1, 0.8, 0, " ", 0.5);
   entry->SetFillColor(ci);
   entry->SetFillStyle(1001);

   ci = TColor::GetColor("#ffcc00");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(2);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   leg->Draw();
   
   TPaveText *pt = new TPaveText(0.53,0.6,0.89,0.85,"brNDC");
   pt->SetBorderSize(0);
   pt->SetFillStyle(0);
   pt->SetTextAlign(11);
   pt->SetTextFont(42);
   TText *pt_LaTex = pt->AddText("ALICE");
   pt_LaTex = pt->AddText("Run 3 pp,#kern[0.4]{#sqrt{#it{s}}} = 13.6 TeV");
   pt_LaTex = pt->AddText("N_{evt} = 1.9 #times 10^{12}");
   pt_LaTex = pt->AddText("#pm 10% global unc. not shown");
   pt->Draw();
   c->Modified();
   c->SetSelected(c);
}
