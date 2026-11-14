"""Функции из предоставленной лабораторной № 2, без исполнения её эксперимента."""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def make_graph(reduced=False):
    g={'l':('input',()), 'a':('pow',('l',2)), 'b':('pow',('l',-1)),
       'c':('scale',('b',2)), 'I1':('add',('a','c')), 's':('offset',('I1',-3)),
       'C10':('param',(0,)), 't1':('mul',('C10','s'))}
    if reduced: g['W']=('copy',('t1',))
    else: g.update({'s2':('pow',('s',2)), 'C20':('param',(1,)),
                   't2':('mul',('C20','s2')), 'W':('add',('t1','t2'))})
    return g

def evaluate(g, lam, theta):
    out={}
    for name,(op,args) in g.items():
        if op=='input': v,d=np.asarray(lam),np.ones_like(lam)
        elif op=='param': v,d=theta[args[0]],0.
        elif op=='pow':
            x,dx=out[args[0]]; p=args[1]; v,d=x**p,p*x**(p-1)*dx
        elif op=='scale':
            x,dx=out[args[0]]; v,d=args[1]*x,args[1]*dx
        elif op=='offset':
            x,dx=out[args[0]]; v,d=x+args[1],dx
        elif op=='copy': v,d=out[args[0]]
        else:
            x,dx=out[args[0]]; y,dy=out[args[1]]
            v,d=(x+y,dx+dy) if op=='add' else (x*y,dx*y+x*dy)
        out[name]=(v,d)
    return out['W'][1]

def design(x,reduced=False):
    g=G1 if reduced else G
    return np.column_stack([evaluate(g,x,t) for t in ([(1.,0.)] if reduced else [(1.,0.),(0.,1.)])])

G=make_graph(); G1=make_graph(True)

def draw_graph(ax,reduced=False):
    g=make_graph(reduced)
    pos={'l':(0,2),'a':(1,2.7),'b':(1,1.3),'c':(2,1.3),'I1':(3,2),'s':(4,2),
         'C10':(4,3.4),'t1':(5,3),'s2':(5,1),'C20':(5,0),'t2':(6,1),'W':(7,2)}
    labels={'l':'λ','a':'λ²','b':'λ⁻¹','c':'×2','I1':'I₁','s':'I₁ − 3','s2':'s²','t1':'C₁₀s','t2':'C₂₀s²'}
    for n,(op,args) in g.items():
        for p in args:
            if isinstance(p,str) and p in g:
                ax.annotate('',xy=pos[n],xytext=pos[p],arrowprops={'arrowstyle':'->','color':'#8797aa','shrinkA':19,'shrinkB':19})
    for n in g:
        ax.text(*pos[n],labels.get(n,n),ha='center',va='center',bbox={'boxstyle':'round,pad=.4','fc':'#ffe2ad' if n.startswith('C') else '#dcebf5','ec':'#678'})
    ax.set(xlim=(-.5,7.5),ylim=(-.6,4));ax.axis('off')

def posterior(A,y,sd):
    p=A.shape[1];m=np.array([.2,0.])[:p];s=np.array([.3,.02])[:p]
    B=A/np.asarray(sd)[:,None];v=y/sd
    H=B.T@B+np.diag(1/s**2);cov=np.linalg.inv(H)
    return cov@(B.T@v+m/s**2),cov

def mh(mu,cov,seed):
    rng=np.random.default_rng(seed);p=len(mu);L=np.linalg.cholesky(cov)
    z=rng.normal(size=(4,p))*3;logp=-.5*(z*z).sum(1);saved=[];accepted=0
    # theta=mu+Lz: целевая плотность z стандартная нормальная.
    for i in range(5000):
        prop=z+rng.normal(size=z.shape)*(2.38/np.sqrt(p));lp=-.5*(prop*prop).sum(1)
        ok=np.log(rng.random(4))<lp-logp;z[ok]=prop[ok];logp[ok]=lp[ok]
        if i>=1000:saved.append(mu+z@L.T);accepted+=ok.sum()
    return np.transpose(saved,(1,0,2)),accepted/(4*4000)

def diagnostics(ch):
    m,n,p=ch.shape;a=np.concatenate([ch[:,:n//2],ch[:,n//2:]],axis=0);nn=a.shape[1]
    W=a.var(1,ddof=1).mean(0);B=nn*a.mean(1).var(0,ddof=1)
    rhat=np.sqrt(((nn-1)/nn*W+B/nn)/W);ess=[]
    for j in range(p):
        z=ch[:,:,j]-ch[:,:,j].mean(1,keepdims=True)
        ft=np.fft.rfft(z,n=2*n,axis=1);ac=np.fft.irfft(ft*ft.conj(),axis=1)[:,:n].real
        ac=(ac/ac[:,0,None]).mean(0);pairs=ac[1:1001].reshape(-1,2).sum(1)
        cut=np.where(pairs<0)[0];pairs=pairs[:cut[0]] if len(cut) else pairs
        ess.append(m*n/max(1,1+2*pairs.sum()))
    return rhat,np.array(ess)
