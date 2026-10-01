#!/usr/bin/env python3
"""Plot spectra and data/model ratios and evaluate ToMCCA covariance chi2."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import ROOT

from utils import (
    build_model_band, build_ratio_graphs, build_central_model_ratio, compute_model_chi2, draw_panel,
    fit_levy, load_config, load_hist, make_error_envelope_graphs,
    open_root_file, print_result, validate_data_histograms,
)


def draw_chi2_panel(results_by_energy, config):
    """Draw reduced chi2 by model, using step histograms to compare energies."""
    style = config['plot']
    models = config['enabled_models']
    tag = '_'.join(str(energy) for energy in results_by_energy)
    canvas = ROOT.TCanvas(f'c_chi2_{tag}', 'Reduced chi2 by model', *style['canvas_size'])
    for side, margin in style['margins'].items():
        getattr(canvas, f'Set{side.capitalize()}Margin')(margin)
    canvas.SetBottomMargin(max(style['margins']['bottom'], 0.18))
    legend = ROOT.TLegend(0.18, 0.72, 0.94, 0.93)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(style['font'])
    legend.SetTextSize(min(style['label_size'], 0.03))
    histograms = []
    n_energies = len(results_by_energy)
    energy_labels = {
        523: '#splitline{B_{#Lambda} = 0.523 #pm 0.013 (stat.) #pm 0.075 (syst.) MeV}{[A1 Collaboration]}',
        102: '#splitline{B_{#Lambda} = 0.102 #pm 0.063 (stat.) #pm 0.067 (syst.) MeV}{[ALICE Collaboration]}',
    }
    maximum = max(result.chi2 / result.ndf
                  for results in results_by_energy.values() for result in results)
    for index, (energy, results) in enumerate(results_by_energy.items()):
        by_name = {result.name: result for result in results}
        hist = ROOT.TH1D(f'h_reduced_chi2_{energy}', ';Model;#chi^{2}/ndf', len(models), 0., len(models))
        hist.SetDirectory(0)
        hist.SetStats(False)
        for bin_index, name in enumerate(models, 1):
            result = by_name[name]
            hist.GetXaxis().SetBinLabel(bin_index, config['models'][name]['label'])
            hist.SetBinContent(bin_index, result.chi2 / result.ndf)
            hist.SetBinError(bin_index, 0.)
        for axis in (hist.GetXaxis(), hist.GetYaxis()):
            axis.SetTitleSize(style['title_size'])
            axis.SetLabelSize(style['label_size'])
            axis.SetTitleFont(style['font'])
            axis.SetLabelFont(style['font'])
        hist.GetXaxis().LabelsOption('h')
        hist.GetYaxis().SetTitleOffset(style['y_title_offset'])
        hist.SetMinimum(0.)
        hist.SetMaximum(max(1., maximum) * 1.6)
        color = (ROOT.kAzure + 1, ROOT.kOrange + 7)[index % 2]
        hist.SetFillColor(color)
        hist.SetLineColor(color)
        if n_energies > 1:
            hist.SetFillStyle(0)
            hist.SetLineWidth(2)
            hist.Draw('HIST' if index == 0 else 'HIST SAME')
        else:
            hist.SetBarWidth(0.8)
            hist.SetBarOffset(0.1)
            hist.Draw('BAR')
        legend.AddEntry(hist, energy_labels.get(energy, f'B_{{#Lambda}} = {energy} keV'),
                        'l' if n_energies > 1 else 'f')
        histograms.append(hist)
    legend.Draw()
    canvas.RedrawAxis()
    return canvas, histograms, legend


def run_analysis(config, *, chi2_only=False):
    """Run with a resolved config from load_config(); return results by energy."""
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetCanvasPreferGL(False)
    spectrum_file = open_root_file(config['input']['spectrum_file'])
    try:
        data_stat = load_hist(spectrum_file, config['input']['stat_hist'])
        data_syst = load_hist(spectrum_file, config['input']['syst_hist'])
    finally:
        spectrum_file.Close()
    validate_data_histograms(data_stat, data_syst)
    for hist in (data_stat, data_syst):
        hist.SetMarkerColor(config['data']['color'])
        hist.SetLineColor(config['data']['color'])
        hist.SetMarkerStyle(config['data']['marker_style'])
        hist.SetMarkerSize(config['data']['marker_size'])
        hist.GetListOfFunctions().Clear()
    fit = None if chi2_only else fit_levy(data_stat, config['levy_fit'])
    evaluation = {
        'use_bin_center': config['evaluation']['mode'] == 'bin_center',
        'n_eval': config['evaluation']['n_eval'],
        'include_data_syst': config['data']['include_syst'],
    }
    model_options = {
        'scale': config['model']['inel_scaling'],
        'norm_rel_unc': config['model']['norm_rel_unc'],
        'fill_alpha': config['model']['fill_alpha'],
        'line_width': config['model']['line_width'],
        'histogram_template': config['input']['model_histogram_template'],
    }
    chi_options = {key: value for key, value in config['chi2'].items()
                   if key not in ('enabled', 'verbose')}
    if chi_options['norm_rel_unc'] is None:
        chi_options['norm_rel_unc'] = config['model']['norm_rel_unc']
    all_results = {}
    model_file = open_root_file(config['input']['model_file'])
    try:
        for energy in config['binding_energies']:
            output_dir = Path(config['output']['directory']) / str(energy)
            output_dir.mkdir(parents=True, exist_ok=True)
            model_graphs, ratio_graphs, results = [], [], []
            print(f'\nBinding energy: {energy} keV')
            for name in config['enabled_models']:
                style = config['models'][name]
                graph = build_model_band(model_file, name, energy, style['color'], **model_options)
                model_graphs.append((graph, style['label']))
                if not chi2_only and config['plot']['ratio']['enabled']:
                    low, high = make_error_envelope_graphs(graph)
                    ratio = build_ratio_graphs(
                        data_stat, data_syst, graph, low, high, style['color'],
                        fill_alpha=config['model']['fill_alpha'],
                        line_width=config['model']['line_width'], **evaluation,
                    )
                    ratio.SetName(f'ratio_{name}_{energy}')
                    ratio_graphs.append((ratio, style['label']))
                integrated_yield = sum(graph.GetY()[i] * (graph.GetErrorXlow(i) + graph.GetErrorXhigh(i))
                                       for i in range(graph.GetN()))
                print(f'ToMCCA ({name}) integrated yield: {integrated_yield:.8g}')
                if config['chi2']['enabled'] or chi2_only:
                    binding_graph = build_model_band(
                        model_file, name, energy, style['color'],
                        **{**model_options, 'norm_rel_unc': 0.},
                    )
                    low, high = make_error_envelope_graphs(binding_graph)
                    result = compute_model_chi2(
                        name, data_stat, data_syst, binding_graph, low, high,
                        **chi_options, **evaluation,
                    )
                    print_result(result, config['chi2']['verbose'])
                    results.append(result)
            all_results[energy] = results
            if results and config['output']['chi2_file']:
                payload = {
                    'binding_energy_keV': energy,
                    'settings': {'chi2': chi_options, 'evaluation': config['evaluation'],
                                 'model': config['model'], 'input': config['input'],
                                 'include_data_syst': config['data']['include_syst']},
                    'results': [asdict(result) for result in results],
                }
                (output_dir / config['output']['chi2_file']).write_text(
                    json.dumps(payload, indent=2, allow_nan=False,
                               default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + '\n'
                )
            if chi2_only:
                continue
            central_ratios = []
            if config['plot']['model_ratio']['enabled']:
                reference_name = config['plot']['model_ratio']['reference_model']
                by_name = dict(zip(config['enabled_models'], (graph for graph, _ in model_graphs)))
                reference = by_name.get(reference_name)
                if reference is None:
                    reference = build_model_band(
                        model_file, reference_name, energy, config['models'][reference_name]['color'],
                        **model_options,
                    )
                for name, (graph, label) in zip(config['enabled_models'], model_graphs):
                    if name != reference_name:
                        central_ratio = build_central_model_ratio(
                            graph, reference, x_range=config['plot']['x_range'],
                            color=config['models'][name]['color'], line_width=config['model']['line_width'],
                        )
                        central_ratios.append((central_ratio, label))
                if not central_ratios:
                    print('Central model ratio: no non-reference models selected; skipping panel.')
            canvases = []
            chi2_histograms = []
            try:
                for panel, graphs in (('spectrum', model_graphs), ('ratio', ratio_graphs), ('model_ratio', central_ratios)):
                    if config['plot'][panel]['enabled'] and graphs:
                        canvas, objects = draw_panel(panel, energy, data_stat, data_syst, graphs, fit, config)
                        canvases.append((canvas, objects))
                        for extension in config['output']['formats']:
                            canvas.SaveAs(str(output_dir / f'{config["output"][panel + "_name"]}.{extension}'))
                if results:
                    canvas, chi2_histograms, legend = draw_chi2_panel({energy: results}, config)
                    canvases.append((canvas, [*chi2_histograms, legend]))
                    for extension in config['output']['formats']:
                        canvas.SaveAs(str(output_dir / f'chi2_models.{extension}'))
                if config['output']['root_file']:
                    output = open_root_file(output_dir / config['output']['root_file'], 'RECREATE')
                    try:
                        output.cd()
                        data_stat.Write()
                        data_syst.Write()
                        if fit:
                            fit.Write('levy')
                        for graph, _ in model_graphs + ratio_graphs + central_ratios:
                            graph.Write()
                        for hist in chi2_histograms:
                            hist.Write()
                        for canvas, _ in canvases:
                            canvas.Write()
                        ROOT.TObjString(json.dumps(config)).Write('analysis_config')
                    finally:
                        output.Close()
            finally:
                for canvas, _ in canvases:
                    canvas.Close()
    finally:
        model_file.Close()
    if not chi2_only and len(all_results) == 2 and all(all_results.values()):
        output_dir = Path(config['output']['directory'])
        canvas, histograms, legend = draw_chi2_panel(all_results, config)
        try:
            for extension in config['output']['formats']:
                canvas.SaveAs(str(output_dir / f'chi2_binding_energies.{extension}'))
            if config['output']['root_file']:
                output = open_root_file(output_dir / config['output']['root_file'], 'RECREATE')
                try:
                    output.cd()
                    for hist in histograms:
                        hist.Write()
                    canvas.Write()
                    ROOT.TObjString(json.dumps(config)).Write('analysis_config')
                finally:
                    output.Close()
        finally:
            canvas.Close()
    return all_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default=str(Path(__file__).with_name('spectrum_config.json')),
                        help='JSON configuration (default: spectrum_config.json beside this script)')
    parser.add_argument('--chi2-only', action='store_true', help='Compute chi2 without fitting or plotting')
    args = parser.parse_args(argv)
    run_analysis(load_config(args.config), chi2_only=args.chi2_only)


if __name__ == '__main__':
    main()
