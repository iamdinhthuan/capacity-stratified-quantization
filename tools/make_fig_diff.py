import json, matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt
import numpy as np

params={'yolo11n':2.6,'yolo11s':9.5,'yolo11m':20.1,'yolo11l':25.3,'yolo11x':57.0,'yolov8s':11.2,'yolov8m':25.9}
# combine sources
b=json.load(open('metrics/sur_bootstrap.json'))           # 5 YOLO11
alld=dict(b)
try:
    old=json.load(open('metrics/sur_bootstrap_int8_ptq.json'))  # has yolov8s
    if 'yolov8s' in old: alld['yolov8s']=old['yolov8s']
except Exception as e: print('no old', e)
try:
    v8=json.load(open('metrics/sur_v8m.json'))
    alld.update(v8)
except Exception as e: print('no v8m', e)

def diff_ci(m):
    v=alld[m]; pt=v['point']['delta_height']; c=v['ci_DIFF_S_minus_XL']
    return pt['S']-pt['XL'], c[0], c[2]
y11=['yolo11n','yolo11s','yolo11m','yolo11l','yolo11x']
v8m=[m for m in ['yolov8s','yolov8m'] if m in alld]
print('available:', list(alld.keys()))

fig,ax=plt.subplots(figsize=(5.4,3.4))
ax.axhspan(-0.07,0,color='0.93',zorder=0)
ax.axhline(0,color='0.4',lw=0.9,ls='--',zorder=1)

xs=[params[m] for m in y11]; pts=[];los=[];his=[]
for m in y11:
    p,lo,hi=diff_ci(m); pts.append(p);los.append(lo);his.append(hi)
pts=np.array(pts);los=np.array(los);his=np.array(his)
err=np.vstack([pts-los,his-pts])
sig=[(lo>0 or hi<0) for lo,hi in zip(los,his)]
ax.plot(xs,pts,color='#1f6feb',lw=1.0,alpha=0.5,zorder=1,label='YOLO11 (primary sweep)')
ax.errorbar(xs,pts,yerr=err,fmt='none',ecolor='#1f6feb',elinewidth=1.6,capsize=3,zorder=2)
for x,p,s in zip(xs,pts,sig):
    ax.scatter([x],[p],s=58,color='#1f6feb' if s else 'white',edgecolor='#1f6feb',
               linewidth=1.7,zorder=3,marker='o')

if v8m:
    xv=[params[m] for m in v8m]; pv=[]
    for m in v8m:
        p,lo,hi=diff_ci(m); pv.append(p)
        ax.errorbar([params[m]],[p],yerr=[[p-lo],[hi-p]],fmt='none',ecolor='#e8730c',
                    elinewidth=1.3,capsize=3,zorder=3)
        ax.scatter([params[m]],[p],s=42,color='white',edgecolor='#e8730c',
                   linewidth=1.6,zorder=4,marker='s')
    ax.plot(xv,pv,color='#e8730c',lw=1.0,alpha=0.4,zorder=1,ls=':',label='YOLOv8 (cross-family)')

ax.set_xscale('log')
ax.set_xlabel('Model parameters (M, log scale)')
ax.set_ylabel(r'$\Delta$AP(S) $-$ $\Delta$AP(XL)')
ax.set_ylim(-0.07,0.11)
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
ax.xaxis.set_major_locator(FixedLocator([3,5,10,20,40]))
ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(['3','5','10','20','40']))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlim(2.2,66)
ax.text(0.03,0.97,'small signs hurt more',transform=ax.transAxes,ha='left',va='top',
        fontsize=7.5,color='0.4',style='italic')
ax.text(0.97,0.035,'large signs hurt more',transform=ax.transAxes,ha='right',va='bottom',
        fontsize=7.5,color='0.4',style='italic')
ax.legend(fontsize=7.5,loc='upper right',framealpha=0.9)
ax.grid(True,alpha=0.25,lw=0.5)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig('figs/fig2_diff_capacity.pdf',bbox_inches='tight')
fig.savefig('figs/fig2_diff_capacity.png',dpi=150,bbox_inches='tight')
print("saved figs/fig2_diff_capacity.pdf/.png")
print("YOLO11 DIFF:", [(m,round(p,3),'SIG' if s else 'ns') for m,p,s in zip(y11,pts,sig)])
