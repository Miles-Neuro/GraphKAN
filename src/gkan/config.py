"""Single-target configuration; paths are relative to the release root."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
VARIANTS=('5-Swish','5-B-spline','5-Identity','1-Swish','w/o KAN','w/o Dyn-Adj','w/o group')
def path(value):
    p=Path(value).expanduser()
    return p if p.is_absolute() else ROOT/p

def cohort(config):
    return list(range(1,21 if config['dataset']['name']=='EEG-Emotion' else 16))

def sessions(config):
    return [1] if config['dataset']['name']=='EEG-Emotion' else [1,2,3]

def read_config(filename):
    c=json.loads(Path(filename).read_text(encoding='utf-8-sig'))
    d,p,m,o,t=(c[k] for k in ('dataset','protocol','model','optimizer','training'))
    if d['name'] not in ('SEED','SEED-IV','EEG-Emotion') or d['format'] not in ('mat_features','npz_manifest'):
        raise ValueError('Unsupported dataset')
    if any(k in d for k in ('subjects','sessions')) or 'seeds' in t or 'session_mode' in p:
        raise ValueError('Batch subject/session/seed configurations are removed. Use one subject/session or one test_subject.')
    if any(k in p for k in ('subjects','sessions','test_subjects','run_subjects')):raise ValueError('Batch targets are not supported')
    if len(d['classes'])!=(4 if d['name']=='SEED-IV' else 3) or len(set(d['classes']))!=len(d['classes']):
        raise ValueError('Invalid dataset classes')
    if p['type']=='SD':
        if type(p.get('subject')) is not int or p['subject'] not in cohort(c) or type(p.get('session')) is not int or p['session'] not in sessions(c):
            raise ValueError('SD requires one valid subject and one session')
        if 'test_subject' in p: raise ValueError('SD uses subject, not test_subject')
        if p['sd_split'] not in ('first_trials','first_per_class','trial_kfold'):raise ValueError('Invalid SD split')
        if p['sd_split']=='trial_kfold' and (type(p.get('folds')) is not int or p['folds']<2):raise ValueError('Invalid trial folds')
    elif p['type']=='SI':
        if type(p.get('test_subject')) is not int or p['test_subject'] not in cohort(c):raise ValueError('SI requires one test_subject')
        if any(k in p for k in ('subject','session','test_subjects')):raise ValueError('SI uses one test_subject and all dataset sessions')
    else:raise ValueError('Protocol must be SD or SI')
    if d['name']=='EEG-Emotion' and d.get('bands')!=[[1,4],[4,8],[8,14],[14,30],[30,48]]:raise ValueError('EEG-Emotion requires gamma 30-48 Hz')
    if c['normalization']!={'scope':'global'}:raise ValueError('All runs require full-cohort, all-session global z-score')
    if t['selection'] not in ('test_peak','final'):raise ValueError('Invalid checkpoint selection')
    if type(t['seed']) is not int or t['seed']<0:raise ValueError('Use one nonnegative seed')
    for v in (m['in_features'],m['hidden_features'],m['head_width'],m['groups'],t['epochs'],t['batch_size']):
        if type(v) is not int or v<1:raise ValueError('Invalid positive integer hyperparameter')
    if t['batch_size']<2 or m['in_features']!=5 or not 0<=m['dropout']<1:raise ValueError('Invalid dimensions/batch/dropout')
    if type(m['chebyshev_order']) is not int or m['chebyshev_order']<1:raise ValueError('Invalid polynomial order')
    v=m['variant']
    if v not in VARIANTS:raise ValueError('Unknown ablation variant')
    if m['groups']!=(1 if v=='1-Swish' else 5):raise ValueError('groups must match variant (w/o group ignores grouping)')
    if m['activation']!=('identity' if v=='5-Identity' else 'bspline' if v=='5-B-spline' else 'swish'):raise ValueError('Activation must match variant')
    if v=='5-B-spline' and m.get('spline')!={'grid_size':5,'degree':3,'range':[-1,1],'grid_update':False}:raise ValueError('Supported spline: fixed cubic basis, 5 intervals, [-1,1]')
    if m['backend']!='kat':raise ValueError('Rational baselines use installed KAT; spline/ungrouped use torch')
    if o['name']!='Adam' or o['lr']<=0 or o['eps']<=0 or len(o['betas'])!=2 or any(not 0<=b<1 for b in o['betas']):raise ValueError('Invalid Adam settings')
    if any(o[k]<0 for k in ('weight_decay','head_weight_decay','rational_weight_decay')):raise ValueError('Negative decay')
    if not 0<=t['label_smoothing']<1:raise ValueError('Invalid label smoothing')
    return c

def write_json(filename,value):
    Path(filename).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
