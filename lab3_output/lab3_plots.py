"""Визуализация результатов ноутбука и исходные рисунки презентации."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from lab3_core import NODES, graph_edges, predict, ALPHA, BETA

COLORS=['#157f86','#f0a44b','#7a65ad']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':130,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':.15,'savefig.facecolor':'white'})


def save(fig,folder,name):
    Path(folder).mkdir(parents=True,exist_ok=True)
    fig.savefig(Path(folder)/(name+'.png'),dpi=180,bbox_inches='tight')
    return fig


def fits(runs,folder='figures'):
    fig,axes=plt.subplots(1,3,figsize=(13.5,3.9))
    for ax,(name,r) in zip(axes,runs.items()):
        x,y=r['x'],r['y'];grid=np.linspace(1,6,240)
        for full,style,color in [(False,'--',COLORS[1]),(True,'-',COLORS[0])]:
            m=r['models'][int(full)]
            ax.plot(grid,predict(grid,m['theta'],full),style,color=color,
                    label='Yeoh-2' if full else 'Yeoh-1',lw=2)
        for flag,marker,color in [('train','o','#607589'),('val','s',COLORS[1]),('test','X','#bd4367')]:
            mask=r[flag];ax.scatter(x[mask],y[mask],marker=marker,c=color,label=flag,s=32,zorder=5)
        ax.set(title=name,xlabel='Растяжение λ',ylabel='Напряжение P, MPa')
    axes[0].legend(fontsize=8);fig.tight_layout()
    return save(fig,folder,'01_fits')


def convergence(runs,folder='figures'):
    fig,axes=plt.subplots(1,2,figsize=(12,3.6))
    for color,(name,r) in zip(COLORS,runs.items()):
        t=r['trace']
        axes[0].plot(t.step,t.p_full,label=name,color=color,lw=2)
        # Из каждой стадии вычитаем её минимальную стоимость: сопоставимый зазор.
        gaps=[]
        for row in t.itertuples():
            costs=[m['score'] for m in r['stages'][row.stage]]
            gaps.append(row.p_full*costs[1]+(1-row.p_full)*costs[0]-min(costs))
        axes[1].plot(t.step,gaps,label=name,color=color,lw=2)
    for ax in axes:
        for step in (120,240):ax.axvline(step,color='#82929b',ls='--',lw=1)
        ax.set_xlabel('Шаг Adam; вертикали — добавление интервенционных сред')
    axes[0].set(ylabel='Вероятность полной структуры',ylim=(-.03,1.03));axes[0].legend(fontsize=8)
    axes[1].set(ylabel='E[J] − min J в текущей стадии',yscale='symlog',ylim=(0,None))
    fig.tight_layout();return save(fig,folder,'02_optimization')


def graph(ax,full,title='',order=None):
    positions={0:(0,1.2),1:(1.7,2.1),2:(3.4,2.1),3:(5,1.2)}
    labels={0:'λ',1:'s',2:'K',3:'P'}
    for i,j in graph_edges(full):
        ax.annotate('',xy=positions[j],xytext=positions[i],
            arrowprops=dict(arrowstyle='->',color=COLORS[0],lw=2,shrinkA=22,shrinkB=22))
    for i,pos in positions.items():
        ax.text(*pos,labels[i],ha='center',va='center',fontsize=13,
            bbox=dict(boxstyle='round,pad=.6',fc='#eef5f5',ec=COLORS[0]))
    ax.text(2.5,.35,'K = C10 + 2·C20·s' if full else 'K = C10 (s не влияет на P)',
            ha='center',fontsize=9,color='#45596c')
    if order:ax.text(2.5,-.2,order.replace('lambda','λ').replace(' > ',' → '),ha='center',fontsize=8)
    ax.set(xlim=(-.7,5.7),ylim=(-.5,2.9),title=title);ax.axis('off')


def graph_sequence(runs,folder='figures'):
    r=runs['Контроль C20=0'];samples=r['samples']
    # Настоящие записи журнала. На старте показываем обе структуры, затем временные срезы.
    inds=[samples[samples.full].index[0],samples[~samples.full].index[0],
          120*16-1,180*16-1,240*16-1,len(samples)-1]
    fig,axes=plt.subplots(2,3,figsize=(13,6.4))
    for ax,idx in zip(axes.ravel(),inds):
        row=samples.loc[idx]
        graph(ax,bool(row.full),f'Запись {idx+1} · шаг {row.step} · '+('Yeoh-2' if row.full else 'Yeoh-1'),row.order)
    fig.suptitle('Реальная последовательность генерации: контроль C20 = 0',fontsize=14)
    fig.tight_layout();return save(fig,folder,'03_graph_sequence')


def interventions(runs,folder='figures'):
    fig,axes=plt.subplots(1,3,figsize=(13.5,3.7))
    for ax,(name,r) in zip(axes,runs.items()):
        for full,color,label in [(False,COLORS[1],'Yeoh-1'),(True,COLORS[0],'Yeoh-2')]:
            d=r['shifts'];a=d[d.full==full].groupby('q').drift.agg(['mean','std'])
            ax.errorbar([0,*a.index],[0,*np.sqrt(a['mean'])],
                        color=color,marker='o',label=label)
            # Разброс показываем по RMS в каждой маске, не sqrt(mean) +/- sqrt(std).
            for q in a.index:
                v=np.sqrt(d[(d.full==full)&(d.q==q)].drift)
                ax.scatter(np.full(len(v),q),v,color=color,alpha=.17,s=12)
        ax.set(title=name,xlabel='Запрошенная доля зануления q',ylabel='RMS сдвига прогноза / σ')
    axes[0].legend(fontsize=8);fig.tight_layout()
    return save(fig,folder,'04_interventions')


def scores(metrics,folder='figures'):
    fig,axes=plt.subplots(1,3,figsize=(13.5,3.5))
    for ax,(name,d) in zip(axes,metrics.groupby('dataset',sort=False)):
        x=np.arange(2);bottom=np.zeros(2)
        for key,weight,label,color in [('accuracy',1,'Точность',COLORS[0]),
                ('complexity',ALPHA,'Сложность',COLORS[2]),('robustness',BETA,'Неустойчивость',COLORS[1])]:
            v=d[key].to_numpy()*weight
            ax.bar(x,v,bottom=bottom,label=label,color=color,width=.55);bottom+=v
        ax.set(xticks=x,xticklabels=['Yeoh-1','Yeoh-2'],title=name,ylabel='J (меньше — лучше)')
    axes[0].legend(fontsize=7);fig.tight_layout();return save(fig,folder,'05_scores')


def causal_semantics(runs,folder='figures'):
    r=runs['Treloar'];m=r['chosen'];x=np.array([1.,2.,3.,4.,5.,6.]);p=predict(x,m['theta'],m['full'])
    fig,axes=plt.subplots(1,2,figsize=(11,3.6))
    graph(axes[0],True,'Структурные уравнения: вмешательство задаёт λ')
    axes[1].plot(x,p,'o-',color=COLORS[0],label='P при do(λ = a), расчёт SCM')
    axes[1].scatter(x,np.zeros(len(x)),marker='x',color='#bd4367',label='do(P_meas = 0): только датчик')
    axes[1].set(xlabel='Заданное растяжение a',ylabel='Напряжение, MPa',title='Предсказание модели, не новый эксперимент')
    axes[1].legend(fontsize=8);fig.tight_layout();return save(fig,folder,'06_causal_semantics')
