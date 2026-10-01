"""Shared ROOT spectrum, ratio, and covariance chi2 helpers.

Importing this module does not open files, fit data, or produce plots.
"""
from dataclasses import dataclass
import copy
import json
from pathlib import Path
import numpy as np
import ROOT


def load_config(filename):
    """Merge a JSON config with the shipped defaults, rejecting unknown keys.

    Relative paths explicitly supplied by a config resolve beside that file;
    inherited paths resolve beside the shipped default config.
    """
    default_path = Path(__file__).with_name('spectrum_config.json')

    def read(path):
        with path.open() as stream:
            config = json.load(stream)
        for section, keys in (
            ('input', ('spectrum_file', 'model_file')),
            ('output', ('directory',)), ('levy_fit', ('header',)),
        ):
            for key in keys:
                if key in config.get(section, {}):
                    value = Path(config[section][key]).expanduser()
                    config[section][key] = str((path.parent / value).resolve())
        return config

    def merge(default, updates, prefix=''):
        merged = copy.deepcopy(default)
        for key, value in updates.items():
            location = f'{prefix}.{key}' if prefix else key
            if key not in default:
                raise ValueError(f'Unknown configuration key: {location}')
            if isinstance(default[key], dict):
                if not isinstance(value, dict):
                    raise ValueError(f'{location} must be an object')
                if location in ('levy_fit.initial_parameters', 'levy_fit.parameter_limits'):
                    merged[key] = value
                else:
                    merged[key] = merge(default[key], value, location)
            else:
                merged[key] = value
        return merged

    config = merge(read(default_path), read(Path(filename).resolve()))
    validate_config(config)
    return config


def validate_config(config):
    def nonnegative(value, name):
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')

    def bounds(value, name):
        if not isinstance(value, list) or len(value) != 2 or not all(np.isfinite(value)) or value[0] >= value[1]:
            raise ValueError(f'{name} must be [min, max] with min < max')

    names = config['enabled_models']
    if not names or len(set(names)) != len(names) or any(n not in config['models'] for n in names):
        raise ValueError('enabled_models must be a nonempty list of unique configured model names')
    energies = config['binding_energies']
    if not energies or any(not isinstance(e, int) or e <= 0 for e in energies) or len(set(energies)) != len(energies):
        raise ValueError('binding_energies must contain unique positive integers')
    evaluation = config['evaluation']
    if evaluation['mode'] not in ('bin_average', 'bin_center'):
        raise ValueError('evaluation.mode must be bin_average or bin_center')
    if not isinstance(evaluation['n_eval'], int) or evaluation['n_eval'] <= 0:
        raise ValueError('evaluation.n_eval must be a positive integer')
    nonnegative(config['model']['norm_rel_unc'], 'model.norm_rel_unc')
    nonnegative(config['model']['inel_scaling'], 'model.inel_scaling')
    if config['model']['inel_scaling'] == 0:
        raise ValueError('model.inel_scaling must be positive')
    if not 0 <= config['model']['fill_alpha'] <= 1:
        raise ValueError('model.fill_alpha must be in [0, 1]')
    chi = config['chi2']
    if chi['norm_rel_unc'] is not None:
        nonnegative(chi['norm_rel_unc'], 'chi2.norm_rel_unc')
    for key in ('norm_corr_fraction', 'binding_corr_fraction'):
        if not 0 <= chi[key] <= 1:
            raise ValueError(f'chi2.{key} must be in [0, 1]')
    if chi['binding_mode'] not in ('average', 'max'):
        raise ValueError('chi2.binding_mode must be average or max')
    for name, value in (('chi2.pt_range', chi['pt_range']), ('levy_fit.range', config['levy_fit']['range'])):
        if value is not None:
            bounds(value, name)
    for key, value in config['levy_fit']['parameter_limits'].items():
        bounds(value, f'levy_fit.parameter_limits.{key}')
    for mapping in ('initial_parameters', 'parameter_limits'):
        if any(key not in ('1', '2', '3') for key in config['levy_fit'][mapping]):
            raise ValueError(f'levy_fit.{mapping} keys must be 1 (n), 2 (C), or 3 (norm)')
    bounds(config['plot']['x_range'], 'plot.x_range')
    if config['plot']['model_ratio']['reference_model'] not in config['models']:
        raise ValueError('plot.model_ratio.reference_model must be a configured model name')
    for name in ('spectrum', 'ratio', 'model_ratio'):
        panel = config['plot'][name]
        if panel['log_y'] and panel['y_min'] <= 0:
            raise ValueError(f'plot.{name}.y_min must be positive for log_y')
        if panel['y_max'] is not None:
            bounds([panel['y_min'], panel['y_max']], f'plot.{name} y range')
    formats = config['output']['formats']
    if any(fmt not in ('pdf', 'png', 'svg', 'eps', 'jpg') for fmt in formats):
        raise ValueError('output.formats supports pdf, png, svg, eps, jpg')
    for key in ('spectrum_name', 'ratio_name', 'model_ratio_name', 'root_file', 'chi2_file'):
        value = config['output'][key]
        if value is None and key in ('root_file', 'chi2_file'):
            continue
        if not isinstance(value, str) or not value or Path(value).name != value or value in ('.', '..'):
            raise ValueError(f'output.{key} must be a filename, without directories')


def open_root_file(path, mode='READ'):
    root_file = ROOT.TFile.Open(str(path), mode)
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f'Cannot open ROOT file: {path}')
    return root_file


def fit_levy(data, config):
    if not config['enabled']:
        return None
    if not Path(config['header']).is_file():
        raise FileNotFoundError(config['header'])
    if not ROOT.gInterpreter.Declare(f'#include "{config["header"]}"'):
        raise RuntimeError('Could not load the LevyTsallis function')
    fit = ROOT.LevyTsallis('levy', config['mass'])
    for index, value in config['initial_parameters'].items():
        fit.SetParameter(int(index), value)
    for index, limits in config['parameter_limits'].items():
        fit.SetParLimits(int(index), *limits)
    fit_range = config['range'] or [data.GetBinLowEdge(1), data.GetXaxis().GetBinUpEdge(data.GetNbinsX())]
    # Keep ownership of the fitted TF1 here; do not attach a clone to the data.
    status = data.Fit(fit, config['options'] + 'NS', '', *fit_range)
    if int(status) != 0:
        raise RuntimeError(f'Levy fit failed with status {int(status)}')
    fit.SetLineColor(config['color'])
    fit.SetLineStyle(config['line_style'])
    fit.SetLineWidth(config['line_width'])
    return fit


def draw_panel(name, energy, data_stat, data_syst, graphs, fit, config):
    """Return a canvas and its Python-owned objects (retain until saved)."""
    style = config['plot']
    panel = style[name]
    x_min, x_max = style['x_range']
    upper_values = [graph.GetY()[i] + graph.GetErrorYhigh(i)
                    for graph, _ in graphs for i in range(graph.GetN())
                    if x_min <= graph.GetX()[i] <= x_max]
    if name == 'spectrum':
        upper_values.extend(data_stat.GetBinContent(i) + np.hypot(
            data_stat.GetBinError(i), data_syst.GetBinError(i) if config['data']['include_syst'] else 0.)
            for i in range(1, data_stat.GetNbinsX() + 1)
            if x_min <= data_stat.GetBinCenter(i) <= x_max)
    y_max = panel['y_max'] if panel['y_max'] is not None else max(
        panel['auto_y_minimum_max'], panel['auto_y_padding'] * max(upper_values, default=0.))
    canvas = ROOT.TCanvas('c' if name == 'spectrum' else f'c_{name}', f'{name}, B_Lambda = {energy} keV', *style['canvas_size'])
    for side, margin in style['margins'].items():
        getattr(canvas, f'Set{side.capitalize()}Margin')(margin)
    canvas.SetLogy(panel['log_y'])
    y_title = panel['y_title']
    if name == 'model_ratio':
        y_title = y_title.replace('{reference}', config['models'][panel['reference_model']]['label'])
    frame = canvas.DrawFrame(x_min, panel['y_min'], x_max, y_max,
                             f';{style["x_title"]};{y_title}')
    for axis in (frame.GetXaxis(), frame.GetYaxis()):
        axis.SetTitleSize(style['title_size'])
        axis.SetLabelSize(style['label_size'])
        axis.SetTitleFont(style['font'])
        axis.SetLabelFont(style['font'])
    frame.GetYaxis().SetTitleOffset(style['y_title_offset'])
    keep = [frame]

    def legend(box, text_size, header=None):
        obj = ROOT.TLegend(*box)
        obj.SetFillStyle(0)
        obj.SetBorderSize(0)
        obj.SetTextFont(style['font'])
        obj.SetMargin(0.1)
        obj.SetTextSize(text_size)
        if header:
            obj.SetHeader(header)
        keep.append(obj)
        return obj

    for graph, _ in graphs:
        graph.Draw(panel['model_draw_option'] + ' SAME')
    if name == 'spectrum':
        if fit:
            fit.Draw('SAME')
        if config['data']['include_syst']:
            data_syst.Draw(panel['syst_draw_option'] + ' SAME')
        data_stat.Draw(panel['stat_draw_option'] + ' SAME')
        leg = legend(panel['data_legend_box'], panel['data_legend_text_size'])
        leg.AddEntry(data_stat, style['collision_label'], 'PE')
        if fit:
            leg.AddEntry(fit, config['levy_fit']['label'], 'L')
        leg.Draw()
    elif panel['unity_line']['enabled']:
        line_style = panel['unity_line']
        line = ROOT.TLine(x_min, 1., x_max, 1.)
        line.SetLineStyle(line_style['style'])
        line.SetLineColor(line_style['color'])
        line.SetLineWidth(line_style['width'])
        line.Draw('SAME')
        keep.append(line)
    leg = legend(panel['model_legend_box'], panel['model_legend_text_size'], panel['legend_header'])
    for graph, label in graphs:
        leg.AddEntry(graph, label, 'L' if name == 'model_ratio' else 'F')
    leg.Draw()
    annotation = ROOT.TPaveText(*panel['annotation_box'], 'NDC')
    annotation.SetBorderSize(0)
    annotation.SetFillStyle(0)
    annotation.SetTextAlign(11)
    annotation.SetTextFont(style['font'])
    annotation.SetTextSize(panel['annotation_text_size'])
    if name == 'model_ratio':
        annotation.AddText('ToMCCA')
        annotation.AddText(f'B_{{#Lambda}} = {energy} keV')
    else:
        annotation.AddText(style['experiment_label'])
    if name == 'ratio':
        annotation.AddText(style['collision_label'])
    annotation.Draw()
    keep.append(annotation)
    canvas.RedrawAxis()
    return canvas, keep


def load_hist(root_file, name):
    hist = root_file.Get(name)
    if not hist or not hist.InheritsFrom('TH1') or hist.GetDimension() != 1:
        raise RuntimeError(f"Missing 1D histogram {name!r} in {root_file.GetName()}")
    hist = hist.Clone()
    hist.SetDirectory(0)
    return hist


def validate_data_histograms(stat, syst):
    if stat.GetNbinsX() != syst.GetNbinsX():
        raise ValueError('Data statistical and systematic histograms have different bin counts')
    for i in range(1, stat.GetNbinsX() + 2):
        if not np.isclose(stat.GetBinLowEdge(i), syst.GetBinLowEdge(i), rtol=0, atol=1e-10):
            raise ValueError('Data statistical and systematic histogram edges differ')


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
    if not isinstance(n_eval, int) or n_eval <= 0:
        raise ValueError('n_eval must be a positive integer')
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


def build_central_model_ratio(model, reference, *, x_range, color, line_width=2):
    """Pointwise central model/reference on the numerator grid, without errors.

    Evaluate the reference by linear interpolation only within its support.
    Absolute normalizations are retained; a flat ratio indicates equal shapes.
    """
    ratio = ROOT.TGraphAsymmErrors()
    ratio.SetName(f'central_ratio_{model.GetName()}_over_{reference.GetName()}')
    ref_x = [reference.GetX()[i] for i in range(reference.GetN())]
    if not ref_x:
        raise ValueError('Cannot form a central ratio with an empty reference')
    for i in range(model.GetN()):
        x = model.GetX()[i]
        if not x_range[0] <= x <= x_range[1]:
            continue
        if not min(ref_x) <= x <= max(ref_x):
            raise ValueError(f'Reference model does not cover pT = {x}')
        denominator = reference.Eval(x)
        numerator = model.GetY()[i]
        if denominator <= 0 or not np.isfinite(denominator) or not np.isfinite(numerator):
            raise ValueError(f'Invalid central model ratio at pT = {x}')
        ratio.SetPoint(ratio.GetN(), x, numerator / denominator)
    if not ratio.GetN():
        raise ValueError('No model points in the selected central-ratio pT range')
    ratio.SetLineColor(color)
    ratio.SetLineWidth(line_width)
    return ratio


def build_ratio_graphs(data_stat, data_syst, model_graph, model_graph_low, model_graph_high, color, *, use_bin_center=False, n_eval=200, fill_alpha=0.2, line_width=2, include_data_syst=True):
    validate_data_histograms(data_stat, data_syst)
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
            use_bin_center=use_bin_center, n_eval=n_eval,
        )
        if model_val <= 0:
            continue
        model_low_val = evaluate_graph_density(
            model_graph_low,
            x_low,
            x_up,
            use_bin_center=use_bin_center, n_eval=n_eval,
        )
        model_high_val = evaluate_graph_density(
            model_graph_high,
            x_low,
            x_up,
            use_bin_center=use_bin_center, n_eval=n_eval,
        )
        if model_low_val <= 0 or model_high_val <= 0:
            raise ValueError(
                f'{model_graph.GetName()}: non-positive model band endpoint in '
                f'pT bin [{x_low}, {x_up}]; cannot draw a finite ratio band'
            )
        data_val = data_stat.GetBinContent(i_bin)
        y = data_val / model_val
        y_data_err = np.hypot(data_stat.GetBinError(i_bin), (data_syst.GetBinError(i_bin) if include_data_syst else 0.)) / model_val
        # Division reverses the model endpoints; retain the full nonlinear shift.
        model_err_low = max(0., y - data_val / model_high_val)
        model_err_high = max(0., data_val / model_low_val - y)
        gr_ratio.SetPoint(i_point, x, y)
        gr_ratio.SetPointEXlow(i_point, x_err)
        gr_ratio.SetPointEXhigh(i_point, x_err)
        gr_ratio.SetPointEYlow(i_point, np.hypot(y_data_err, model_err_low))
        gr_ratio.SetPointEYhigh(i_point, np.hypot(y_data_err, model_err_high))
        i_point += 1

    gr_ratio.SetFillColorAlpha(color, fill_alpha)
    gr_ratio.SetLineColor(color)
    gr_ratio.SetLineWidth(line_width)

    return gr_ratio


def build_model_band(model_file, model_name, binding_energy, color, *,
                     scale=0.75, norm_rel_unc=0.155, fill_alpha=0.2, line_width=2,
                     histogram_template='Hypertriton_Wigner:{model}_{energy}{variation}_pt'):
    """Build the plotted band; use norm_rel_unc=0 for the binding-only band."""
    histograms = []
    for variation in ('', 'Low', 'Hi'):
        key = histogram_template.format(model=model_name, energy=binding_energy, variation=variation)
        hist = model_file.Get(key)
        if not hist or not hist.InheritsFrom('TH1') or hist.GetDimension() != 1:
            raise RuntimeError(f'Missing model histogram: {key}')
        histograms.append(hist)
    mid, low, high = histograms
    validate_data_histograms(mid, low)
    validate_data_histograms(mid, high)

    # The new models already share a 0.1 GeV/c binning, including HWH.
    graph = ROOT.TGraphAsymmErrors(mid)
    graph.SetName(f'{model_name}_{binding_energy}')
    for i_bin in range(1, mid.GetNbinsX() + 1):
        y_mid = mid.GetBinContent(i_bin) * scale
        y_low = low.GetBinContent(i_bin) * scale
        y_high = high.GetBinContent(i_bin) * scale
        model_unc = norm_rel_unc * y_mid
        i_point = i_bin - 1
        graph.SetPoint(i_point, mid.GetBinCenter(i_bin), y_mid)
        graph.SetPointEXlow(i_point, 0.5 * mid.GetBinWidth(i_bin))
        graph.SetPointEXhigh(i_point, 0.5 * mid.GetBinWidth(i_bin))
        graph.SetPointEYlow(i_point, np.hypot(max(0., y_mid - y_low), model_unc))
        graph.SetPointEYhigh(i_point, np.hypot(max(0., y_high - y_mid), model_unc))

    graph.SetFillColorAlpha(color, fill_alpha)
    graph.SetLineColor(color)
    graph.SetLineWidth(line_width)
    return graph



@dataclass
class Chi2Result:
    name: str
    n_points: int
    chi2: float
    ndf: int
    p_value: float
    chi2_data_only: float
    p_value_data_only: float
    chi2_corr_up_data_only: float
    p_value_corr_up_data_only: float
    nuisance_pulls: np.ndarray
    pt_min: np.ndarray
    pt_max: np.ndarray
    residuals: np.ndarray
    chi2_contrib: np.ndarray
    chi2_diag_contrib: np.ndarray
    data_values: np.ndarray
    model_values: np.ndarray
    data_unc: np.ndarray
    norm_unc: np.ndarray
    binding_unc: np.ndarray
    corr_unc: np.ndarray



def split_uncertainty_source(unc, corr_fraction):
    if not np.isfinite(corr_fraction) or corr_fraction < 0.0 or corr_fraction > 1.0:
        raise ValueError(f"Correlation fraction must be in [0, 1], got {corr_fraction}")
    corr_unc = np.sqrt(corr_fraction) * unc
    uncorr_unc = np.sqrt(1.0 - corr_fraction) * unc
    return corr_unc, uncorr_unc


def covariance_chi2(residuals, data_unc, sources):
    cov = np.diag(data_unc * data_unc)
    for source in sources:
        cov += np.outer(source, source)
    cov_inv = np.linalg.pinv(cov, hermitian=True)
    weighted_residuals = cov_inv @ residuals
    chi2_contrib = residuals * weighted_residuals
    return float(np.sum(chi2_contrib)), chi2_contrib, cov


def profiled_nuisance_pulls(residuals, data_unc, sources):
    source_matrix = np.column_stack(sources)
    data_weight = np.diag(1.0 / (data_unc * data_unc))
    lhs = source_matrix.T @ data_weight @ source_matrix + np.eye(source_matrix.shape[1])
    rhs = source_matrix.T @ data_weight @ residuals
    return np.linalg.solve(lhs, rhs)



def compute_model_chi2(
    name,
    data_stat,
    data_syst,
    model_graph,
    model_low_graph,
    model_high_graph,
    *,
    norm_rel_unc=0.155,
    binding_mode="average",
    norm_corr_fraction=1.0,
    binding_corr_fraction=1.0,
    use_bin_center=False,
    n_eval=200,
    include_data_syst=True,
    pt_range=None,
):
    """Covariance chi2 for a fixed prediction (ndf = number of selected bins).

    Supply scaled central and binding-only endpoint graphs: normalization is
    added separately here. Correlation fractions refer to variance. Data
    stat/syst errors are independent between bins, as in the original script.
    pt_range selects fully contained bins; None uses all bins.
    """
    if norm_rel_unc < 0 or not np.isfinite(norm_rel_unc):
        raise ValueError("norm_rel_unc must be finite and nonnegative")
    validate_data_histograms(data_stat, data_syst)
    data = np.array(
        [data_stat.GetBinContent(i_bin) for i_bin in range(1, data_stat.GetNbinsX() + 1)],
        dtype=float,
    )
    pt_min = np.array(
        [data_stat.GetBinLowEdge(i_bin) for i_bin in range(1, data_stat.GetNbinsX() + 1)],
        dtype=float,
    )
    pt_max = np.array(
        [
            data_stat.GetBinLowEdge(i_bin) + data_stat.GetBinWidth(i_bin)
            for i_bin in range(1, data_stat.GetNbinsX() + 1)
        ],
        dtype=float,
    )
    stat_unc = np.array(
        [data_stat.GetBinError(i_bin) for i_bin in range(1, data_stat.GetNbinsX() + 1)],
        dtype=float,
    )
    syst_unc = np.array(
        [data_syst.GetBinError(i_bin) for i_bin in range(1, data_syst.GetNbinsX() + 1)],
        dtype=float,
    )
    data_unc = np.hypot(stat_unc, syst_unc if include_data_syst else 0.)

    model, model_low, model_up = (
        np.array([evaluate_graph_density(graph, low, high, use_bin_center, n_eval)
                  for low, high in zip(pt_min, pt_max)])
        for graph in (model_graph, model_low_graph, model_high_graph)
    )
    residuals = data - model
    norm_unc = norm_rel_unc * model

    binding_up = np.maximum(model_up - model, 0.0)
    binding_low = np.maximum(model - model_low, 0.0)
    if binding_mode == "average":
        binding_unc = 0.5 * (binding_up + binding_low)
    elif binding_mode == "max":
        binding_unc = np.maximum(binding_up, binding_low)
    else:
        raise ValueError(f"Unsupported binding uncertainty mode '{binding_mode}'")

    norm_corr_unc, norm_uncorr_unc = split_uncertainty_source(norm_unc, norm_corr_fraction)
    binding_corr_unc, binding_uncorr_unc = split_uncertainty_source(binding_unc, binding_corr_fraction)
    fit_data_unc = np.sqrt(
        data_unc * data_unc
        + norm_uncorr_unc * norm_uncorr_unc
        + binding_uncorr_unc * binding_uncorr_unc
    )

    corr_unc = np.sqrt(norm_unc * norm_unc + binding_unc * binding_unc)
    total_diag_unc = np.sqrt(data_unc * data_unc + corr_unc * corr_unc)
    chi2_diag_contrib = np.divide(
        residuals * residuals,
        total_diag_unc * total_diag_unc,
        out=np.zeros_like(residuals),
        where=total_diag_unc > 0.0,
    )

    valid = (data_unc > 0.0) & (fit_data_unc > 0.0) & (model > 0.0)
    valid &= np.isfinite(data) & np.isfinite(model) & np.isfinite(total_diag_unc)
    if pt_range is not None:
        valid &= (pt_min >= pt_range[0]) & (pt_max <= pt_range[1])
    if not np.any(valid):
        raise ValueError(f"{name}: no valid bins selected for chi2")
    if not np.all(valid):
        skipped = np.nonzero(~valid)[0] + 1
        print(f"{name}: skipping invalid or unselected bins {skipped.tolist()}")


    chi2, chi2_contrib, _ = covariance_chi2(
        residuals[valid],
        fit_data_unc[valid],
        (norm_corr_unc[valid], binding_corr_unc[valid]),
    )
    nuisance_pulls = profiled_nuisance_pulls(
        residuals[valid],
        fit_data_unc[valid],
        (norm_corr_unc[valid], binding_corr_unc[valid]),
    )
    n_points = int(np.count_nonzero(valid))
    p_value = float(ROOT.Math.chisquared_cdf_c(chi2, n_points))

    data_only_valid = valid & (data_unc > 0.0)
    n_data_only_points = int(np.count_nonzero(data_only_valid))
    chi2_data_only = float(np.sum((residuals[data_only_valid] / data_unc[data_only_valid]) ** 2))
    p_value_data_only = float(ROOT.Math.chisquared_cdf_c(chi2_data_only, n_data_only_points))
    shifted_up_model = model + corr_unc
    shifted_up_residuals = data - shifted_up_model
    chi2_corr_up_data_only = float(np.sum((shifted_up_residuals[data_only_valid] / data_unc[data_only_valid]) ** 2))
    p_value_corr_up_data_only = float(ROOT.Math.chisquared_cdf_c(chi2_corr_up_data_only, n_data_only_points))
    return Chi2Result(
        name=name,
        n_points=n_points,
        chi2=chi2,
        ndf=n_points,
        p_value=p_value,
        chi2_data_only=chi2_data_only,
        p_value_data_only=p_value_data_only,
        chi2_corr_up_data_only=chi2_corr_up_data_only,
        p_value_corr_up_data_only=p_value_corr_up_data_only,
        nuisance_pulls=nuisance_pulls,
        pt_min=pt_min[valid],
        pt_max=pt_max[valid],
        residuals=residuals[valid],
        chi2_contrib=chi2_contrib,
        chi2_diag_contrib=chi2_diag_contrib[valid],
        data_values=data[valid],
        model_values=model[valid],
        data_unc=data_unc[valid],
        norm_unc=norm_unc[valid],
        binding_unc=binding_unc[valid],
        corr_unc=corr_unc[valid],
    )


def print_result(result, verbose):
    print(f"\n{result.name}")
    print(f"  chi2 / ndf = {result.chi2:.4f} / {result.ndf}")
    print(f"  chi2/ndf   = {result.chi2 / result.ndf:.4f}")
    print(f"  p-value    = {result.p_value:.6g}")
    print(f"  fit prob.  = {100.0 * result.p_value:.3f}%")
    print(f"  data-only chi2/ndf (central model) = {result.chi2_data_only:.4f} / {result.ndf}, fit prob. = {100.0 * result.p_value_data_only:.3f}%")
    print(f"  data-only chi2/ndf (+1 sigma total model up) = {result.chi2_corr_up_data_only:.4f} / {result.ndf}, fit prob. = {100.0 * result.p_value_corr_up_data_only:.3f}%")
    print(f"  profiled norm pull = {result.nuisance_pulls[0]:.3f} sigma")
    print(f"  profiled binding pull = {result.nuisance_pulls[1]:.3f} sigma")

    if not verbose:
        return
    print("  bin-by-bin contributions:")
    for i, (pt_min, pt_max, data, model, residual, chi2_contrib, chi2_diag_contrib, data_unc, norm_unc, binding_unc, corr_unc) in enumerate(
        zip(
            result.pt_min,
            result.pt_max,
            result.data_values,
            result.model_values,
            result.residuals,
            result.chi2_contrib,
            result.chi2_diag_contrib,
            result.data_unc,
            result.norm_unc,
            result.binding_unc,
            result.corr_unc,
        ),
        start=1,
    ):
        line = f"    {i} [{pt_min:.2f}, {pt_max:.2f}]: chi2_i_cov={chi2_contrib:.6f}, chi2_i_diag={chi2_diag_contrib:.6f}"
        if verbose:
            line += (
                f", data={data:.6e}, model={model:.6e}, residual={residual:.6e}, "
                f"data_unc={data_unc:.6e}, norm_unc={norm_unc:.6e}, "
                f"binding_unc={binding_unc:.6e}, corr_unc={corr_unc:.6e}"
            )
        print(line)
