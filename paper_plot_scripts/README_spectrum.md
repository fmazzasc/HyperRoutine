# Spectrum and ToMCCA comparison

Run in an environment with PyROOT and NumPy:

```bash
python3 spectrum.py spectrum_config.json
python3 spectrum.py spectrum_config.json --chi2-only
```

Omitting the config uses `spectrum_config.json` beside the script. The old
`tomcca_chi2.py` is now a thin entry point for the second command and accepts the
same JSON config; its former individual command-line flags are replaced by
config fields. Importing either script does not run an analysis.

The supplied config lists the available settings. A custom JSON file can contain
just overrides; nested objects merge with the supplied defaults. Parameter maps
`levy_fit.initial_parameters` and `parameter_limits` replace the entire map, so
`{}` removes all configured initial values or limits. Unknown keys are errors.
Explicit relative input/output/header paths resolve relative to the custom JSON
file; inherited paths remain relative to the supplied default config.

For example, a small config for standard Congleton only:

```json
{
  "enabled_models": ["Congleton"],
  "binding_energies": [102],
  "output": {"directory": "congleton_only"},
  "chi2": {"binding_mode": "max", "verbose": true}
}
```

| Section | Settings |
| --- | --- |
| `input` | Data/model ROOT files, data histogram names, model histogram name template (`model`, `energy`, `variation`; variations are empty, `Low`, `Hi`) |
| `output` | Output directory, image formats and names, ROOT and chi2 JSON filenames; `null` disables either ROOT or JSON output |
| `binding_energies`, `enabled_models` | Energies and models to process; available models are `Congleton`, `CongletonHwH`, `Gaussian` |
| `models` | Legend labels and ROOT integer color codes |
| `model` | Inelastic scaling, relative normalization uncertainty, fill opacity and line width |
| `evaluation` | `bin_average` or `bin_center`, integration sample count; shared by ratios and chi2 |
| `data` | Include systematic errors, marker style/size and color |
| `levy_fit` | Enable, C++ header, mass, fit range/options, initial parameters/limits, line style and label; indices 1/2/3 are n/C/normalization |
| `chi2` | Enable, normalization uncertainty override, normalization/binding correlation fractions, `average`/`max` binding symmetrisation, pT range and verbose output |
| `plot` | Canvas size, margins, font and axis sizes/titles, pT display range, experiment/collision labels |
| `plot.spectrum`, `plot.ratio` | Enable panels, log scale, y range and automatic scaling, draw options, annotation/legend positions and text sizes; ratio unity line styling |

Set `y_max` to `null` for automatic scaling; otherwise supply an explicit upper
limit. A logarithmic panel needs positive `y_min`. Display ranges do not select
chi2 bins: use `chi2.pt_range` for that (only fully contained bins are included).
`levy_fit.range` is also independent; `null` fits the entire data range.

Each energy gets a subdirectory containing plots, ROOT objects and
`chi2_results.json` with chi2, ndf, p-value, nuisance pulls, bin contributions and
effective analysis settings. The ROOT file also stores the resolved config.
The supplied selection preserves both Congleton variants and excludes Gaussian.

When chi2 is enabled, each energy also gets `chi2_models.pdf/png`, showing
chi2/ndf with the enabled models on the x-axis. Runs with two binding energies
add `chi2_binding_energies.pdf/png` in the output directory, with two overlaid,
unfilled step histograms. The energy legends include the statistical and
systematic measurement uncertainties and collaboration names for 102 and
523 keV, expressed in MeV. These plots follow `output.formats`; their histograms
and canvases are saved in the configured ROOT filename (per-energy files for
individual plots, and a file in the output directory for the comparison).
`--chi2-only` skips all plots and ROOT output as before.

`plot.model_ratio` adds `ratio_models_central.pdf/png`: central predictions of
each enabled model divided by `reference_model` (default `Congleton`) versus
pT. It retains absolute normalizations and draws no uncertainty bands or nuisance
shifts. A constant ratio means the two predictions have the same shape. Ratios
use model-bin centres and interpolate the reference, independently of data-bin
averaging. The reference can be disabled in the other panels; it is loaded when
needed here. If only the reference is selected, this panel is skipped. Set
`plot.model_ratio.enabled` to `false` to disable it; its axes, styling and output
name (`output.model_ratio_name`) are configurable like the other panels.

## Statistical conventions

`utils.compute_model_chi2` retains the covariance calculation from the original
script. Statistical and systematic data errors are combined in quadrature and
treated as independent between pT bins. Normalization and binding uncertainty
are separate sources. Correlation fractions specify **fractions of variance**:
0 means independent bins; 1 means a fully correlated source. Binding uncertainty
is symmetrised by either averaging the up/down magnitudes or taking their maximum.

The chi2 now reads the same `HypertritonHistograms.root` predictions as the plots,
including each energy's own H.H. variation histograms. It uses the same graph
interpolation and bin averaging/centre evaluation as the ratio. The old script
instead used histogram overlap averages from `output_tomcca.root`, with special
H.H. rebinning. Those old-file transformations are no longer needed.

`chi2.norm_rel_unc: null` inherits `model.norm_rel_unc` (currently 0.155).
Set it to `0.0` to retain the old standalone script's zero normalization
uncertainty. The ratio still uses the plotted uncertainty, irrespective of this
chi2-only override. The central prediction is fixed: ndf is the number of valid
selected bins, with no subtraction for constrained nuisance pulls or the separate
Lévy fit. Reported p-values use the symmetrised Gaussian covariance model.

Ratio errors use exact divisions by the asymmetric model endpoints, with data
errors added in quadrature. The chi2 covariance and the displayed ratio band
therefore serve different purposes; bin correlations affect chi2, not the band.
Nonpositive ratio denominators raise an error rather than show a finite bound.
Invalid chi2 bins (including zero data uncertainty) are reported and excluded.
The legacy `corr_unc` result field contains the full symmetrised model magnitude;
the extra shifted-model diagnostic uses that magnitude, not just its correlated
portion. It is not the profiled best-fit prediction.

## Calling the chi2 function

Pass the scaled central model graph and **binding-only** endpoint graphs;
normalization uncertainty is added separately to avoid counting it twice:

```python
from utils import build_model_band, make_error_envelope_graphs, compute_model_chi2

binding = build_model_band(model_file, "Congleton", 102, 632,
                           scale=0.75, norm_rel_unc=0.0)
low, high = make_error_envelope_graphs(binding)
result = compute_model_chi2(
    "Congleton", data_stat, data_syst, binding, low, high,
    norm_rel_unc=0.155, binding_mode="average",
    norm_corr_fraction=1.0, binding_corr_fraction=1.0,
    use_bin_center=False, n_eval=200,
    include_data_syst=True, pt_range=None,
)
print(result.chi2, result.ndf, result.p_value)
```

For the complete workflow from Python, use
`spectrum.run_analysis(utils.load_config("my_config.json"))`.

Numerical tests: `python3 -m unittest test_spectrum`.
