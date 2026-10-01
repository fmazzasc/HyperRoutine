"""Numerical checks without the analysis input files: python3 -m unittest test_spectrum."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import ROOT

from utils import (build_ratio_graphs, build_central_model_ratio, compute_model_chi2, evaluate_graph_density,
                   load_config)


class SpectrumTests(unittest.TestCase):
    def setUp(self):
        self.stat = ROOT.TH1D('stat', '', 2, 0., 2.)
        self.syst = ROOT.TH1D('syst', '', 2, 0., 2.)
        for hist in (self.stat, self.syst):
            hist.SetDirectory(0)
        for i, value in enumerate((8., 12.), 1):
            self.stat.SetBinContent(i, value)
            self.stat.SetBinError(i, 1.)
            self.syst.SetBinError(i, 0.)
        self.mid = self.flat_graph(10.)
        self.low = self.flat_graph(8.)
        self.high = self.flat_graph(12.)

    @staticmethod
    def flat_graph(value):
        graph = ROOT.TGraph(2)
        graph.SetPoint(0, 0., value)
        graph.SetPoint(1, 2., value)
        return graph

    def compute(self, **kwargs):
        return compute_model_chi2('toy', self.stat, self.syst, self.mid, self.low,
                                  self.high, norm_rel_unc=0., **kwargs)

    def test_correlated_shift_cannot_absorb_opposite_residuals(self):
        result = self.compute(binding_corr_fraction=1.)
        # A common model shift cannot explain residuals (-2, +2).
        self.assertAlmostEqual(result.chi2, 8.)
        np.testing.assert_allclose(result.nuisance_pulls, [0., 0.], atol=1e-12)
        self.assertEqual(result.ndf, 2)

    def test_uncorrelated_and_partial_covariance(self):
        for fraction in (0., 0.3, 1.):
            result = self.compute(binding_corr_fraction=fraction)
            residual = np.array([-2., 2.])
            covariance = np.diag([1. + 4. * (1. - fraction)] * 2) + 4. * fraction * np.ones((2, 2))
            expected = residual @ np.linalg.solve(covariance, residual)
            self.assertAlmostEqual(result.chi2, expected)

    def test_binding_symmetrisation_and_selection(self):
        self.high = self.flat_graph(14.)
        average = self.compute(binding_mode='average', pt_range=[0., 1.])
        maximum = self.compute(binding_mode='max', pt_range=[0., 1.])
        self.assertAlmostEqual(average.chi2, 4. / 10.)
        self.assertAlmostEqual(maximum.chi2, 4. / 17.)
        self.assertEqual(average.ndf, 1)
        with self.assertRaisesRegex(ValueError, 'no valid bins'):
            self.compute(pt_range=[3., 4.])

    def test_exact_ratio_endpoints(self):
        ratio = build_ratio_graphs(self.stat, self.syst, self.mid, self.low, self.high, ROOT.kRed)
        self.assertAlmostEqual(ratio.GetY()[0], 0.8)
        self.assertAlmostEqual(ratio.GetErrorYhigh(0), np.hypot(0.1, 8. / 8. - 0.8))
        self.assertAlmostEqual(ratio.GetErrorYlow(0), np.hypot(0.1, 0.8 - 8. / 12.))
        with self.assertRaisesRegex(ValueError, 'non-positive'):
            build_ratio_graphs(self.stat, self.syst, self.mid, self.flat_graph(0.), self.high, ROOT.kRed)

    def test_bin_average_and_bin_center(self):
        triangle = ROOT.TGraph(3)
        for i, (x, y) in enumerate(((0., 1.), (1., 3.), (2., 1.))):
            triangle.SetPoint(i, x, y)
        self.assertAlmostEqual(evaluate_graph_density(triangle, 0., 2.), 2.)
        self.assertAlmostEqual(evaluate_graph_density(triangle, 0., 2., True), 3.)

    def test_central_model_ratio_retains_normalization_and_shape(self):
        numerator = ROOT.TGraph(3)
        for i, (x, y) in enumerate(((0., 20.), (1., 30.), (2., 40.))):
            numerator.SetPoint(i, x, y)
        ratio = build_central_model_ratio(numerator, self.mid, x_range=[0., 2.], color=632)
        np.testing.assert_allclose([ratio.GetY()[i] for i in range(3)], [2., 3., 4.])
        self.assertTrue(all(ratio.GetErrorYhigh(i) == 0 for i in range(3)))

    def test_config_partial_paths_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.json'
            path.write_text(json.dumps({'enabled_models': ['Gaussian'], 'output': {'directory': 'plots'},
                                        'levy_fit': {'parameter_limits': {'2': [0.01, 1.]}}}))
            config = load_config(path)
            self.assertEqual(config['enabled_models'], ['Gaussian'])
            self.assertEqual(config['output']['directory'], str(Path(directory) / 'plots'))
            self.assertEqual(config['levy_fit']['parameter_limits'], {'2': [0.01, 1.]})
            self.assertTrue(Path(config['input']['model_file']).is_absolute())
            for invalid in ({'chi2': {'typo': 1}}, {'evaluation': {'mode': 'invalid'}},
                            {'chi2': {'norm_corr_fraction': 2}}, {'enabled_models': []}):
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == '__main__':
    unittest.main()
