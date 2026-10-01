import copy,json,sys,tempfile,unittest,importlib.util
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import BSpline
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gkan.config import read_config,cohort,sessions
from gkan.data import Record,distance_adjacency,pooled_statistics,load_records
from gkan.splits import concatenate,make_folds
from gkan.model import GraphKAN,supports
from gkan.activations import GroupBSpline
from gkan.engine import make_optimizer,seed_all
from gkan.preprocessing import extract_de
from gkan.eeglab import interval

def synthetic(c):
    rng=np.random.default_rng(5);rows=[]
    n=24 if c['dataset']['name']=='SEED-IV' else 16 if c['dataset']['name']=='EEG-Emotion' else 15
    for subject in cohort(c):
        for session in sessions(c):
            rows.append(Record(subject,session,rng.normal(size=(n*2,62,5)).astype(np.float32),np.repeat(np.arange(n)%len(c['dataset']['classes']),2),np.repeat(np.arange(1,n+1),2),'fixture','fixture'))
    return concatenate(rows)

class Tests(unittest.TestCase):
    def config(self,name='seed_sd'):return read_config(ROOT/'Configs'/f'{name}.json')
    def test_all_configs(self):
        files=list((ROOT/'Configs').rglob('*.json'));count=0
        for p in files:
            if p.name.startswith('preprocess'):continue
            c=read_config(p);self.assertEqual(c['training']['epochs'],100);count+=1
        self.assertEqual(count,13)
    def test_reject_batch_and_no_normalization(self):
        c=self.config()
        changes=[('dataset','subjects',[1,2]),('training','seeds',[123,456]),('normalization','scope','none'),('protocol','subject',[1,2]),('protocol','session',[1,2])]
        for section,k,v in changes:
            cc=copy.deepcopy(c);cc[section][k]=v
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'c.json';p.write_text(json.dumps(cc))
                with self.assertRaises((ValueError,TypeError)):read_config(p)
    def test_single_sd(self):
        for name in ['seed_sd','seed_iv_sd','eeg_emotion_sd']:
            c=self.config(name);c['protocol']['subject']=7;d=synthetic(c);fs=make_folds(d,c)
            self.assertEqual(len(fs),5 if name=='eeg_emotion_sd' else 1)
            for f in fs:
                self.assertEqual(set(d['subject'][f.train]),{7});self.assertEqual(set(d['subject'][f.test]),{7})
                self.assertEqual(set(d['session'][f.train]),{1});self.assertFalse(set(d['group'][f.train])&set(d['group'][f.test]))
            if name=='seed_sd':
                self.assertEqual(set(d['trial'][fs[0].train]),set(range(1,10)))
            if name=='eeg_emotion_sd':
                ids=np.concatenate([f.test for f in fs]);self.assertEqual(len(ids),len(set(ids)));self.assertEqual(len(ids),32)
    def test_single_si_all_sessions_and_peers(self):
        for name in ['seed_si','seed_iv_si','eeg_emotion_si']:
            c=self.config(name);c['protocol']['test_subject']=5;d=synthetic(c);fs=make_folds(d,c)
            self.assertEqual(len(fs),1);f=fs[0]
            self.assertEqual(set(d['subject'][f.train]),set(cohort(c))-{5});self.assertEqual(set(d['subject'][f.test]),{5})
            self.assertEqual(set(d['session'][f.train]),set(sessions(c)));self.assertEqual(set(d['session'][f.test]),set(sessions(c)))
    def test_global_stats(self):
        x=np.arange(120,dtype=np.float32).reshape(4,6,5);a,n=pooled_statistics([x[:2],x[2:]])
        self.assertEqual(n,4);np.testing.assert_allclose(a['mean'],x.mean(0,keepdims=True));np.testing.assert_allclose(a['scale'],x.std(0,keepdims=True))
    def test_spline_basis_gradient(self):
        m=GroupBSpline().double();x=torch.linspace(-.99,.99,31,dtype=torch.double)
        np.testing.assert_allclose(m.basis(x).detach(),BSpline(m.grid.numpy(),np.eye(8),3)(x.numpy()),atol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(m,(torch.randn(2,2,5,dtype=torch.double,requires_grad=True)*.2,)))
    def test_eeg_manifest_and_band_validation(self):
        c=self.config('eeg_emotion_sd')
        manifest=json.loads((ROOT/'examples/eeg_emotion_features.example.json').read_text())
        self.assertEqual([r['subject'] for r in manifest['records']],list(range(1,21)))
        with tempfile.TemporaryDirectory() as td:
            c['dataset']['root']=td
            for record in manifest['records']:
                np.savez(Path(td)/record['file'],x=np.zeros((6,62,5),np.float32),y=np.repeat(np.arange(3),2),trial=np.arange(1,7))
            path=Path(td)/'manifest.json';path.write_text(json.dumps(manifest))
            self.assertEqual(load_records(c)[0].subject,1)
            c['protocol']['subject']=20
            self.assertEqual(load_records(c)[0].subject,20)
            self.assertEqual(len(load_records(c,normalization_pool=True)),20)
            del manifest['bands'];path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load_records(c)
    def test_de_and_events(self):
        t=np.arange(600)/200;x=extract_de(np.array([np.sin(2*np.pi*10*t)*1e-5]),200,[[1,4],[4,8],[8,14],[14,30],[30,48]])
        self.assertTrue(np.isfinite(x).all());self.assertGreater(x[:,:,2].mean(),x[:,:,4].mean())
        self.assertEqual(interval(dict(pnts=20,event=[dict(type='S17',latency=2),dict(type='S33',latency=12)]),'S17','S33'),(1,11))
    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("kat_rational") is not None,'CUDA required for installed KAT')
    def test_variants_gradient_frozen_graph_and_reload(self):
        for file in (ROOT/'Configs/ablations').glob('*.json'):
            c=read_config(file);seed_all(123);m=GraphKAN(distance_adjacency(c),c['model'],3).cuda();o,_=make_optimizer(m,c)
            before=m.g_adj_weight.detach().clone();x=torch.randn(6,62,5,device='cuda');y=torch.arange(6,device='cuda')%3
            torch.nn.functional.cross_entropy(m(x),y).backward()
            for n,p in m.named_parameters():
                if p.requires_grad:self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all(),n)
                else:self.assertIsNone(p.grad)
            o.step()
            if c['model']['variant']=='w/o Dyn-Adj':torch.testing.assert_close(m.g_adj_weight,before,rtol=0,atol=0)
            m.eval();out=m(x).detach();restored=GraphKAN(distance_adjacency(c),c['model'],3).cuda().eval();restored.load_state_dict(m.state_dict());torch.testing.assert_close(restored(x),out,rtol=0,atol=0)
            for g in o.param_groups:
                for p in g['params']:self.assertTrue(p.requires_grad)
    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("kat_rational") is not None,'CUDA required')
    def test_ungrouped_initial_equivalence_and_independence(self):
        c=self.config();a=distance_adjacency(c);seed_all(123);base=GraphKAN(a,c['model'],3).cuda().eval()
        cc=copy.deepcopy(c);cc['model']['variant']='w/o group';seed_all(123);edge=GraphKAN(a,cc['model'],3).cuda().eval()
        x=torch.randn(2,62,5,device='cuda');torch.testing.assert_close(base(x),edge(x),atol=2e-5,rtol=2e-5)
        coeff=edge.gcn.gc1[0].lin.kan.weight_numerator
        self.assertEqual(tuple(coeff.shape),(5,32,6));old=coeff.detach().clone()
        with torch.no_grad():coeff[0,0,0]+=1
        self.assertEqual(int((coeff!=old).sum()),1)

if __name__=='__main__':unittest.main()
