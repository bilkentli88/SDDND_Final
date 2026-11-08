"""Algebra, gradient and accounting checks; fixtures are not study outcomes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import e2_data as data
import e2_neural as neural
import run_experiment2 as run

HERE=Path(__file__).resolve().parent


def fixture_rows(config):
    rows=[]
    for sigma in config['noise_levels']:
        for design in config['designs']:
            for seed in config['data_seeds']:
                for method in run.DIRECT:
                    row={'design':design,'noise_std':sigma,'data_seed':seed,
                         'optimization_seed':None,'method':method}
                    row.update(status='ok',delay_rmse=2.) if design=='informative' else row.update(status='abstain_structural')
                    rows.append(row)
                for opt in config['optimization_seeds']:
                    for method in run.NEURAL:
                        row={'design':design,'noise_std':sigma,'data_seed':seed,
                             'optimization_seed':opt,'method':method}
                        if design=='degenerate' and method in ('A1','A2'):
                            row.update(status='abstain_structural')
                        else:
                            row.update(status='ok',delay_rmse=1. if method=='P3' else 2.,
                                       final_parameters_hash='shared' if design=='degenerate' else method)
                        rows.append(row)
    return rows


class Validation(unittest.TestCase):
    def setUp(self):
        self.c=run.load_config(HERE/'protocol.json')

    def test_history_norm_and_closed_form_recovery(self):
        c=np.asarray(self.c['history_c']);d=np.asarray(self.c['history_d'])
        self.assertEqual(c.mean(),0.);self.assertEqual(d.mean(),0.);self.assertEqual(c@d,0.)
        for state in (.45,.83,1.2):
            for delay in (.15,.237,.321,.5):
                q=data.signatures(state,[delay],'informative',self.c)[0]
                w=q-q.mean()
                expected=2*np.sqrt(2)*self.c['history_amplitude']*np.sin(self.c['history_frequency']*delay/2)
                self.assertAlmostEqual(np.linalg.norm(w),expected,places=13)
                b=-self.c['model']['growth']*state/self.c['model']['capacity']
                response=self.c['model']['growth']*state+b*q
                y=response-response.mean()
                C,D=c@y/2,d@y/2
                recovered=2/self.c['history_frequency']*np.arctan(-D/C)
                self.assertAlmostEqual(recovered,delay,places=12)

    def test_profile_matches_derived_squared_distance(self):
        for state in (.45,.83,1.2):
            true=float(data.tau_true(state,self.c))
            response=data.initial_responses(state,'informative',self.c)
            _,eligible,p,_=data.profile(response,state,'informative',self.c)
            self.assertTrue(eligible)
            grid=np.linspace(*self.c['delay_bounds'],self.c['delay_profile_points'])
            q=data.signatures(state,[true],'informative',self.c)[0]
            b=-self.c['model']['growth']*state/self.c['model']['capacity']
            norm=np.linalg.norm(q-q.mean())
            exact=(abs(b)*norm*np.sin(self.c['history_frequency']*(grid-true)/2))**2/4
            np.testing.assert_allclose(p,exact,rtol=1e-11,atol=1e-16)

    def test_degenerate_profiles_abstain(self):
        _,valid,p,norm=data.profile(np.array([-.4,.2,.3,.1]),.7,'degenerate',self.c)
        self.assertFalse(valid);self.assertEqual(norm,0.)
        self.assertTrue(np.array_equal(p,np.full_like(p,p[0])))
        anchor=data.estimate_anchors(np.ones((16,21)),self.c['train_states'],'degenerate',self.c)
        self.assertFalse(anchor['eligible'].any());self.assertTrue(np.isnan(anchor['delays']).all())

    def test_positive_history_and_analytic_solution_box(self):
        theta=np.linspace(-.5,0,1001)
        meta=data.metadata(self.c['train_states'],'informative',self.c)
        h=data.history(theta[None,:],meta['s'][:,None],meta['c'][:,None],meta['d'][:,None],self.c)
        self.assertGreaterEqual(float(h.min()),.27-1e-14)
        maximum=max(1.38,1.2*np.exp(1.1))
        minimum=.45*np.exp(1.1*(1-maximum/1.5))
        self.assertGreater(minimum,0.)
        self.assertGreater(1.1*minimum/1.5,.05)

    def test_noise_pairing_preserves_initial_state(self):
        clean=np.ones((16,21))
        low=data.add_noise(clean,.0005,900007)
        high=data.add_noise(clean,.0015,900007)
        np.testing.assert_array_equal(low[:,0],clean[:,0])
        np.testing.assert_allclose((low-clean)/.0005,(high-clean)/.0015,rtol=0,atol=1e-12)

    def test_reference_zero_growth_fixture(self):
        c=copy.deepcopy(self.c);c['model']['growth']=0.
        meta=data.metadata([.7],'informative',c)
        result=data.simulate_reference(meta,.0025,c)
        np.testing.assert_array_equal(result,np.full((4,21),.7))

    def test_reference_initial_segment_against_independent_ODE(self):
        from scipy.integrate import solve_ivp
        c=copy.deepcopy(self.c);c['horizon']=.15
        meta=data.metadata([.71],'informative',c)
        s,cv,dv=meta['s'],meta['c'],meta['d']
        def rhs(t,x):
            query=t-data.tau_true(x,c)
            assert (query<0).all()
            delayed=s+c['history_amplitude']*(cv*np.sin(c['history_frequency']*query)+dv*(1-np.cos(c['history_frequency']*query)))
            return c['model']['growth']*x*(1-delayed/c['model']['capacity'])
        times=np.arange(4)*c['observation_step']
        times[-1]=c['horizon']
        independent=solve_ivp(rhs,(0.,c['horizon']),s,method='DOP853',t_eval=times,rtol=1e-12,atol=1e-13)
        self.assertTrue(independent.success)
        reference=data.simulate_reference(meta,.0025,c)
        np.testing.assert_allclose(reference,independent.y,rtol=1e-10,atol=1e-11)

    def test_neural_gain_prior_and_delay_bounds(self):
        neural.set_seed(900003)
        model=neural.RestrictedModel(self.c)
        x=torch.linspace(.45,1.2,11).reshape(-1,1)
        with torch.no_grad():
            gain=model.derivative(x,torch.ones_like(x))-model.derivative(x,torch.zeros_like(x))
            delay=model.tau(x)
        self.assertTrue((gain<-.05).all())
        self.assertTrue((delay>.15).all() and (delay<.5).all())

    def test_rollout_delay_gradient_against_finite_difference(self):
        config=self.c
        class Probe:
            def __init__(self,bias):self.bias=bias
            def tau(self,x):return .28+.07*torch.tanh(2*(x-.8))+self.bias
            def derivative(self,x,past):return 1.1*x*(1-past/1.5)
        meta=neural.tensors(data.metadata([.7],'informative',config),dtype=torch.float64)
        bias=torch.tensor(.003,dtype=torch.float64,requires_grad=True)
        value=neural.rollout(Probe(bias),meta,10,config)[0,-1]
        analytic=float(torch.autograd.grad(value,bias)[0])
        eps=1e-6
        a=float(neural.rollout(Probe(torch.tensor(.003+eps,dtype=torch.float64)),meta,10,config)[0,-1])
        b=float(neural.rollout(Probe(torch.tensor(.003-eps,dtype=torch.float64)),meta,10,config)[0,-1])
        numeric=(a-b)/(2*eps)
        self.assertGreater(abs(analytic),1e-8)
        self.assertAlmostEqual(analytic,numeric,delta=max(1e-7,abs(numeric)*1e-4))

    def test_full_accounting_and_structural_abstentions(self):
        c=copy.deepcopy(self.c);c['execution_mode']='full'
        with tempfile.TemporaryDirectory() as folder:
            summary=run.summarize(fixture_rows(c),c,Path(folder))
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['trained_neural_fits'],180)
        self.assertEqual(summary['direct_functions'],20)
        self.assertEqual(summary['structural_abstention_records'],80)
        self.assertEqual(summary['degenerate_P3_P4_pairs_equal'],30)
        self.assertTrue(summary['consistent_practical_joint_advantage_at_both_noise_levels'])

    def test_missing_fit_withholds_complete_case_verdict(self):
        c=copy.deepcopy(self.c);c['execution_mode']='full'
        rows=fixture_rows(c)
        removed=next(r for r in rows if r['method']=='P3' and r['design']=='informative')
        rows.remove(removed)
        with tempfile.TemporaryDirectory() as folder:
            summary=run.summarize(rows,c,Path(folder))
        self.assertFalse(summary['complete'])
        self.assertIsNone(summary['consistent_practical_joint_advantage_at_both_noise_levels'])
        self.assertTrue(all(level['consistent_practical_joint_advantage'] is None for level in summary['per_noise'].values()))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Validation))
    print('Tests use algebraic checks and labelled fixtures; no new noisy learning outcomes are produced.')
    raise SystemExit(0 if result.wasSuccessful() else 1)
