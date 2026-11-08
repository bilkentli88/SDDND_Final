"""Preparation checks using archived observations and labelled synthetic fixtures.

No new experimental seed is evaluated here. Neural execution is checked by
the notebook smoke run, because Torch is unavailable in the preparation runtime.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import run_experiment1 as runner

HERE = Path(__file__).resolve().parent


def fixture_rows(config):
    rows = []
    for seed in config['data_seeds']:
        for opt in config['optimization_seeds']:
            for method, value in [('P3',1.0),('P4',4.0),('A1',2.0),('A2',2.5)]:
                rows.append({'data_seed':seed,'optimization_seed':opt,'method':method,
                             'status':'ok','delay_rmse':value})
        for method,value in [('Linear',2.0),('PCHIP',3.0)]:
            rows.append({'data_seed':seed,'optimization_seed':None,'method':method,
                         'status':'ok','delay_rmse':value})
    return rows


class AccountingChecks(unittest.TestCase):
    def setUp(self):
        self.config = runner.load_config(HERE/'protocol.json')

    def test_fixed_counts_and_dataset_units(self):
        with tempfile.TemporaryDirectory() as folder:
            result = runner.summarize(fixture_rows(self.config),self.config,Path(folder))
            self.assertTrue(result['complete'])
            self.assertEqual(result['recorded_neural_fits'],60)
            self.assertEqual(result['recorded_direct_functions'],10)
            self.assertEqual(result['independent_noise_datasets'],5)
            self.assertEqual(result['dataset_comparisons']['A1']['complete_datasets'],5)
            self.assertEqual(result['dataset_comparisons']['A1']['median_dataset_ratio'],0.5)
            self.assertTrue(result['consistent_practical_joint_advantage'])

    def test_failed_fit_withholds_primary_verdict(self):
        rows = fixture_rows(self.config)
        failed = next(r for r in rows if r['method'] == 'A1')
        failed.update(status='failed')
        failed.pop('delay_rmse')
        with tempfile.TemporaryDirectory() as folder:
            result = runner.summarize(rows,self.config,Path(folder))
            self.assertFalse(result['complete'])
            self.assertIsNone(result['consistent_practical_joint_advantage'])
            self.assertEqual(result['failed_fits'],1)
            self.assertEqual(result['dataset_comparisons']['A1']['complete_datasets'],4)

    def test_missing_fit_withholds_primary_verdict(self):
        rows = fixture_rows(self.config)
        rows.pop(0)
        with tempfile.TemporaryDirectory() as folder:
            result = runner.summarize(rows,self.config,Path(folder))
            self.assertFalse(result['complete'])
            self.assertIsNone(result['consistent_practical_joint_advantage'])

    def test_zero_denominator_withholds_ratio_verdict(self):
        rows = fixture_rows(self.config)
        for row in rows:
            if row['method'] == 'A1':
                row['delay_rmse'] = 0.0
        with tempfile.TemporaryDirectory() as folder:
            result = runner.summarize(rows,self.config,Path(folder))
            self.assertTrue(result['complete'])
            self.assertIsNone(result['consistent_practical_joint_advantage'])

    def test_dataset_median_precedes_ratio(self):
        rows = fixture_rows(self.config)
        seed = self.config['data_seeds'][0]
        for method,values in [('P3',[1.,3.,100.]),('A1',[2.,8.,101.])]:
            group = [r for r in rows if r['data_seed']==seed and r['method']==method]
            for row,value in zip(group,values):
                row['delay_rmse'] = value
        with tempfile.TemporaryDirectory() as folder:
            runner.summarize(rows,self.config,Path(folder))
            import pandas as pd
            pairs = pd.read_csv(Path(folder)/'dataset_pairs.csv')
            ratio = pairs[(pairs.data_seed==seed)&(pairs.comparator=='A1')].ratio.iloc[0]
            self.assertAlmostEqual(ratio,3./8.)

    def test_direct_baselines_do_not_extrapolate(self):
        states = np.array([-0.75,-0.25,0.25,0.75])
        anchors = np.array([0.2,0.25,0.31,0.36])
        linear,pchip = runner.direct_curves(states,states,anchors)
        self.assertTrue(np.allclose(linear,anchors))
        self.assertTrue(np.allclose(pchip,anchors))
        _,outside = runner.direct_curves(np.array([-1.,1.]),states,anchors)
        self.assertTrue(np.isnan(outside).all())

    def test_notebook_cells_have_valid_python_syntax(self):
        notebook = json.loads((HERE/'ASOC_Experiment1_Colab.ipynb').read_text())
        self.assertEqual(notebook['nbformat'],4)
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':
                self.assertEqual(cell['execution_count'],None)
                self.assertEqual(cell['outputs'],[])
                compile(''.join(cell['source']),f'notebook_cell_{i}','exec')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True,type=Path)
    args = parser.parse_args()
    config = runner.load_config(HERE/'protocol.json')
    data,_ = runner.load_data_module(args.repo,config)
    checks = runner.verify_archived_anchors(args.repo,config,data)
    print(f'PASS: {len(checks)} archived datasets reproduce anchor MAEs, exact histories and finite interpolants.',flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AccountingChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    torch_present = importlib.util.find_spec('torch') is not None
    print('Torch installed in this runtime:',torch_present)
    print('New neural training performed by this validation: NO.')
    print('Synthetic fixture values used only to verify accounting; they are not experimental results.')


if __name__=='__main__':
    main()
