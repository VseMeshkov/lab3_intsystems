"""Воспроизводимый эксперимент: дифференцируемый поиск порядка и выбор SCM.

Вычисления на NumPy, SciPy и pandas.
"""
from pathlib import Path
from itertools import permutations
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import norm, ks_2samp
from lab2_core import design, posterior, mh, diagnostics

NODES = ('lambda', 's', 'K', 'P')
ORDERS = list(permutations(range(4)))
ALPHA, BETA, TAU = .10, .05, .05


def graph_edges(full):
    """Механизмы: s=s(lambda), K=C10+2*C20*s, P=2*(lambda-lambda^-2)*K."""
    return [(0, 1), (0, 3), (2, 3)] + ([(1, 2)] if full else [])


def compatible(order, full):
    rank = np.argsort(order)
    return all(rank[i] < rank[j] for i, j in graph_edges(full))


STATES = [(order, full) for order in ORDERS for full in (False, True)
          if compatible(order, full)]
# Оценки порядка: чем больше r_i, тем позже ставится переменная i.
FEATURES = np.array([list(np.argsort(o) / 3) + [float(f)] for o, f in STATES])
BASE_LOG = np.array([-np.log(sum(f == full for _, f in STATES)) for _, full in STATES])


def datasets():
    real = np.array([[1.12885809997723,.13605622522522354],
        [1.41047907112889,.3339561891891877],[1.8940384427749,.5194874054054027],
        [2.45295475981396,.6802811261261245],[3.06286792207369,.8658123423423396],
        [3.62246723257553,1.0637123063063016],[4.05525742348847,1.2368747747747637],
        [4.81998243731095,1.5955684594594526],[5.40781214427424,1.9542621441441415],
        [5.79324812176797,2.3005870810810753]])
    rng = np.random.default_rng(42)
    x = np.linspace(1.1,6,30)
    out = {}
    for name, c20 in [('Контроль C20=0',0.),('Контроль C20=0.002',.002)]:
        out[name] = (x, design(x) @ [.2,c20] + rng.normal(0,.03,len(x)), .03)
    out['Treloar'] = (real[:,0],real[:,1],.08)
    return out


def split(n):
    """Test сохраняется как в ЛР2. Validation выделяется из прежнего train."""
    test = np.arange(n) % 4 == 3
    pool = np.flatnonzero(~test)
    val = np.zeros(n,dtype=bool)
    val[pool[np.arange(len(pool)) % 3 == 2]] = True
    return ~test & ~val, val, test


def fit(x, y, sigma, full):
    return posterior(design(x, reduced=not full), y, np.full(len(y),sigma))


def predict(x, theta, full):
    return design(np.asarray(x), reduced=not full) @ theta


def physical(theta, full):
    grid=np.linspace(1,6,501)
    return bool(theta[0]>0 and np.all(np.diff(predict(grid,theta,full))>0))


def score_models(x, y, sigma, train, val, levels=(.25,.5), repeats=12):
    """Все маски только на train. Одинаковые маски для двух структур."""
    models=[]; shifts=[]
    for full in (False,True):
        mu,cov=fit(x[train],y[train],sigma,full)
        pv=predict(x[val],mu,full)
        accuracy=float(np.mean(((pv-y[val])/sigma)**2))
        stability=[]
        for seed in range(repeats):
            perm=np.random.default_rng(2026+seed).permutation(train.sum())
            for q in levels:
                yy=y[train].copy();yy[perm[:int(q*len(yy))]]=0
                m,c=fit(x[train],yy,sigma,full)
                drift=float(np.mean(((predict(x[val],m,full)-pv)/sigma)**2))
                stability.append(drift)
                shifts.append(dict(full=full,mask=seed,q=q,actual_q=int(q*len(yy))/len(yy),
                                   drift=drift,C10=m[0],C20=m[1] if full else 0.))
        robust=float(np.mean(stability)) if stability else 0.
        models.append(dict(full=full,theta=mu,cov=cov,accuracy=accuracy,
            complexity=len(mu),robustness=robust,
            val_rmse=float(np.sqrt(accuracy)*sigma),physical=physical(mu,full),
            score=accuracy+ALPHA*len(mu)+BETA*robust))
    return models,pd.DataFrame(shifts)


def objective(z, costs, tau=TAU):
    """Точная конечная сумма, аналитический градиент по оценкам порядка и ветви."""
    logits=BASE_LOG+FEATURES@z
    q=softmax(logits)
    logq=np.log(q+1e-300)
    # KL к базе с равными массами структур: число сортировок не даёт преимущества.
    logbase=BASE_LOG-np.log(2.)
    terms=costs+tau*(logq-logbase+1)
    loss=float(q@costs+tau*(q@(logq-logbase))+.001*(z@z))
    grad=FEATURES.T@(q*(terms-q@terms))+.002*z
    return loss,grad,q


def optimize(stages, seed=11, steps=120, draws=16):
    """Накопление сред интервенций -> обновление распределения -> новые DAG."""
    rng=np.random.default_rng(seed)
    z=rng.normal(0,.03,5);moment=np.zeros(5);variance=np.zeros(5)
    trace=[]; samples=[];step=0
    for stage,models in enumerate(stages):
        costs=np.array([models[int(full)]['score'] for _,full in STATES])
        for iteration in range(steps):
            step+=1
            loss,grad,prob=objective(z,costs)
            moment=.9*moment+.1*grad;variance=.999*variance+.001*grad**2
            z-=.045*(moment/(1-.9**step))/(np.sqrt(variance/(1-.999**step))+1e-8)
            _,_,prob=objective(z,costs)
            # Вероятности не являются байесовским posterior: это распределение поиска.
            trace.append(dict(step=step,stage=stage,loss=objective(z,costs)[0],
                p_full=sum(prob[k] for k,(_,f) in enumerate(STATES) if f),
                entropy=float(-prob@np.log(prob+1e-300)),
                **{f'r{i}':float(z[i]) for i in range(4)},branch_logit=float(z[4])))
            for k in rng.choice(len(STATES),size=draws,p=prob):
                o,f=STATES[k]
                samples.append(dict(step=step,stage=stage,state=int(k),full=f,
                    order=' > '.join(NODES[i] for i in o),
                    edges=json.dumps([[NODES[i],NODES[j]] for i,j in graph_edges(f)]),
                    probability=float(prob[k]),score=float(costs[k])))
    return pd.DataFrame(trace),pd.DataFrame(samples),z


def lab2_rule(models):
    small,large=models
    mu,cov=large['theta'],large['cov']
    lo,hi=mu[1]+np.array([-1,1])*norm.ppf(.975)*np.sqrt(cov[1,1])
    remove=bool(lo<=0<=hi and small['val_rmse']-large['val_rmse']<=.02
                and small['physical'])
    return remove,float(lo),float(hi)


def run(out_dir='results'):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    summaries=[];tables=[];runs={};diags=[];ksrows=[]
    for k,(name,(x,y,sd)) in enumerate(datasets().items()):
        train,val,test=split(len(x))
        stages=[score_models(x,y,sd,train,val,levels=l)[0]
                for l in [(),(.25,),(.25,.5)]]
        models,shifts=score_models(x,y,sd,train,val)
        trace,samples,z=optimize(stages,seed=11+k)
        remove,lo,hi=lab2_rule(models)
        # Выбираем только среди действительно сгенерированных структур.
        seen=set(samples.full.astype(bool))
        eligible=[m for m in models if m['full'] in seen and m['physical']
                  and (m['full'] or remove)]
        if not eligible:
            raise RuntimeError('Нет допустимого графа среди сгенерированных: '+name)
        chosen=min(eligible,key=lambda m:m['score'])
        baseline=min(models,key=lambda m:m['accuracy']+ALPHA*m['complexity'])
        full=chosen['full']
        # Test впервые используется после фиксации структуры и всех настроек.
        # Для сопоставимости оставляем те же параметры, оцененные только на train.
        test_rmse=float(np.sqrt(np.mean((predict(x[test],chosen['theta'],full)-y[test])**2)))
        for m in models:
            tables.append(dict(dataset=name,**{a:b for a,b in m.items() if a not in ('theta','cov')},
                               C10=m['theta'][0],C20=m['theta'][1] if m['full'] else 0.))
        summaries.append(dict(dataset=name,n_train=int(train.sum()),n_val=int(val.sum()),
            n_test=int(test.sum()),chosen='Yeoh-2' if full else 'Yeoh-1',
            ml_baseline='Yeoh-2' if baseline['full'] else 'Yeoh-1',C20_low=lo,C20_high=hi,
            removal_allowed=remove,val_rmse=chosen['val_rmse'],test_rmse=test_rmse,
            score=chosen['score'],unique_orders=samples.order.nunique(),
            sampled_graphs=len(samples),unique_structures=samples.full.nunique()))
        # MH из ЛР2: контроль распределений параметров. KS p-value только на iid.
        chains=[]
        for q in (0.,.5):
            yy=y[train].copy()
            perm=np.random.default_rng(2026).permutation(len(yy))
            yy[perm[:int(q*len(yy))]]=0
            mu,cov=fit(x[train],yy,sd,True)
            ch,acc=mh(mu,cov,seed=800+10*k+int(q*10))
            rh,es=diagnostics(ch)
            diags.append(dict(dataset=name,q=q,acceptance=acc,rhat=rh.max(),ess=es.min(),
                              mean_error_sd=float(np.max(abs(ch.mean((0,1))-mu)/np.sqrt(cov.diagonal())))))
            chains.append((mu,cov,ch))
        rng=np.random.default_rng(900+k)
        for j,par in enumerate(('C10','C20')):
            m0,c0,ch0=chains[0];m1,c1,ch1=chains[1]
            testks=ks_2samp(rng.normal(m0[j],np.sqrt(c0[j,j]),500),
                              rng.normal(m1[j],np.sqrt(c1[j,j]),500))
            ksrows.append(dict(dataset=name,parameter=par,
                D_MH=ks_2samp(ch0[:,:,j].ravel(),ch1[:,:,j].ravel()).statistic,
                p_iid=testks.pvalue,p_bonferroni=min(1.,6*testks.pvalue)))
        prefix=f'dataset_{k}'
        trace.to_csv(out/f'{prefix}_optimization.csv',index=False)
        samples.to_csv(out/f'{prefix}_graphs.csv',index=False)
        shifts.to_csv(out/f'{prefix}_interventions.csv',index=False)
        runs[name]=dict(x=x,y=y,sigma=sd,train=train,val=val,test=test,
            stages=stages,models=models,chosen=chosen,trace=trace,samples=samples,z=z,shifts=shifts)
    summary=pd.DataFrame(summaries);metrics=pd.DataFrame(tables)
    diagnostic=pd.DataFrame(diags);ks=pd.DataFrame(ksrows)
    for name,table in [('summary',summary),('candidate_metrics',metrics),('mh_diagnostics',diagnostic),('ks',ks)]:
        table.to_csv(out/f'{name}.csv',index=False)
    return runs,summary,metrics,diagnostic,ks
