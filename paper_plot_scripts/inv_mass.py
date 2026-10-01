import ROOT
ROOT.gROOT.SetBatch(True)

REMOVE_PAVE_NAMES = ["title"]
REMOVE_PAVE_CLASSES = ["TPaveText"]
TEXT_SIZE_PIXELS = 28
X_AXIS_RANGE = (2.96, 3.034)

signal_extraction_file = ROOT.TFile("/home/fmazzasc/run3/results/2024_bdt/signal_extraction_pp2024_sigscan_newlumi.root")

cv = signal_extraction_file.Get("pt_bin_4/fits_sig_dscb_bkg_pol2/data_frame_fit_ptbin_eff_88")
cv2 = signal_extraction_file.Get("pt_bin_5/fits_sig_dscb_bkg_pol2/data_frame_fit_ptbin_eff_88")


if not cv or not cv2:
    raise RuntimeError("Could not retrieve one or both RooPlot objects from the ROOT file")


def add_alice_pave(pad):
    pad.cd()
    pinfo_alice = ROOT.TPaveText(0.18, 0.6, 0.42, 0.85, "NDC")
    pinfo_alice.SetBorderSize(0)
    pinfo_alice.SetFillStyle(0)
    pinfo_alice.SetTextAlign(11)
    pinfo_alice.SetTextFont(42)
    pinfo_alice.AddText("ALICE")
    pinfo_alice.AddText("pp, #sqrt{#it{s}} = 13.6 TeV")
    pinfo_alice.AddText("{}^{3}_{#bar{#Lambda}}#bar{H} #rightarrow ^{3}#bar{He} + #pi^{+}")
    pinfo_alice.Draw()
    return pinfo_alice


def inspect_and_prune_paves(pad, pad_label):
    pad.Update()
    primitives = pad.GetListOfPrimitives()
    to_remove = []

    print(f"--- {pad_label} ---")
    for obj in primitives:
        if obj.InheritsFrom("TPave"):
            print(f"TPave name={obj.GetName()} class={obj.ClassName()}")
            if obj.GetName() in REMOVE_PAVE_NAMES and obj.ClassName() in REMOVE_PAVE_CLASSES:
                print(f"Marked for removal: name={obj.GetName()} class={obj.ClassName()}")
                to_remove.append(obj)

    for obj in to_remove:
        print(f"Removing name={obj.GetName()} class={obj.ClassName()}")
        primitives.Remove(obj)

    pad.Modified()
    pad.Update()


def style_pad_primitives(pad):
    primitives = pad.GetListOfPrimitives()

    for obj in primitives:
        if obj.InheritsFrom("TPave") or obj.InheritsFrom("TLegend"):
            obj.SetTextSizePixels(TEXT_SIZE_PIXELS)

    pad.Modified()
    pad.Update()


def build_single_canvas(name, title, frame, left_margin, right_margin):
    canvas = ROOT.TCanvas(name, title, 800, 700)
    canvas.cd()
    ROOT.gPad.SetMargin(left_margin, right_margin, 0.15, 0.08)
    frame.Draw()
    canvas.Update()
    inspect_and_prune_paves(canvas, name)
    style_pad_primitives(canvas)
    return canvas


cv.GetXaxis().SetRangeUser(*X_AXIS_RANGE)
cv2.GetXaxis().SetRangeUser(*X_AXIS_RANGE)

multi_pad = ROOT.TCanvas("multi_pad", "Invariant mass fits", 1600, 700)
multi_pad.Divide(2, 1)

multi_pad.cd(1)
ROOT.gPad.SetMargin(0.15, 0.02, 0.15, 0.08)
cv.Draw()

multi_pad.cd(2)
ROOT.gPad.SetMargin(0.03, 0.05, 0.15, 0.08)
cv2.Draw()

multi_pad.Update()

inspect_and_prune_paves(multi_pad.GetPad(1), "pad 1")
inspect_and_prune_paves(multi_pad.GetPad(2), "pad 2")
style_pad_primitives(multi_pad.GetPad(1))
style_pad_primitives(multi_pad.GetPad(2))
# alice_pave = add_alice_pave(multi_pad.GetPad(1))
multi_pad.GetPad(1).Modified()
multi_pad.GetPad(1).Update()

single_pad_1 = build_single_canvas("single_pad_1", "Invariant mass fit pt bin 4", cv, 0.15, 0.05)
single_pad_2 = build_single_canvas("single_pad_2", "Invariant mass fit pt bin 5", cv2, 0.15, 0.05)

output_file = ROOT.TFile("invariant_mass_fits.root", "RECREATE")
multi_pad.Write()
single_pad_1.Write()
single_pad_2.Write()
output_file.Close()
