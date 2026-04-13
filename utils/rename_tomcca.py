import ROOT

input_file = ROOT.TFile('ToMCCAHypertritonALICEPaper.root')
## list all the keys in the file
print(input_file.GetListOfKeys())
h_clones = []
## some objects have same name, we just access them without caring about the name, and clone them to avoid issues with ownership
for i,key in enumerate(input_file.GetListOfKeys()):
    obj = key.ReadObj()
    obj.SetDirectory(0)
    clone = obj.Clone()
    if i == 0:
        clone.SetName("congleton_low")
    elif i == 1:
        clone.SetName("congleton")
    elif i == 2:
        clone.SetName("congleton_up")
    elif i == 3:
        clone.SetName("gaussian")
    elif i == 4:
        clone.SetName("gaussian_up")
    elif i == 5:
        clone.SetName("gaussian_low")

    h_clones.append(clone)

file2 = ROOT.TFile('ToMCCALH3_HwHTune.root')
for i,key in enumerate(file2.GetListOfKeys()):
    obj = key.ReadObj()
    obj.SetDirectory(0)
    clone = obj.Clone()
    if i == 0:
        clone.SetName("congleton_hwh")
    else:
        continue
    h_clones.append(clone)
output_file = ROOT.TFile('output_tomcca.root', 'RECREATE')
for h in h_clones:
    h.Write()
output_file.Close()