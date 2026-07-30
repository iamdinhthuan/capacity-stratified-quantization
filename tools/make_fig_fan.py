import json, matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype']=42   # TrueType, not Type 3 (submission-safe)
matplotlib.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

params={'yolo11n':2.6,'yolo11s':9.5,'yolo11m':20.1,'yolo11l':25.3,'yolo11x':57.0}
order=['yolo11n','yolo11s','yolo11m','yolo11l','yolo11x']
xs=[params[m] for m in order]
bins=['XS','S','M','L','XL']
colors={'XS':'0.6','S':'#9ecae1','M':'#6baed6','L':'#3182bd','XL':'#08519c'}

absloss={b:[] for b in bins}; relloss={b:[] for b in bins}
for m in order:
    f=json.load(open(f'metrics/{m}_fp32.json')); i=json.load(open(f'metrics/{m}_int8_ptq.json'))
    for b in bins:
        f32=f['height_bin_ap'][b]['mAP50-95']; i8=i['height_bin_ap'][b]['mAP50-95']
        d=f32-i8
        absloss[b].append(d*100)
        relloss[b].append(100*d/f32 if f32>1e-6 else float('nan'))

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8.2,3.4))
def style(ax):
    ax.set_xscale('log'); ax.axhline(0,color='0.5',lw=0.7,ls=':')
    ax.xaxis.set_major_locator(FixedLocator([3,5,10,20,40]))
    ax.xaxis.set_major_formatter(FixedFormatter(['3','5','10','20','40']))
    ax.xaxis.set_minor_locator(NullLocator()); ax.set_xlim(2.2,66)
    ax.set_xlabel('Model parameters (M, log scale)')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.grid(True,alpha=0.25,lw=0.5)
for b in bins:
    dashed = (b=='XS')
    lbl = 'XS ($n{=}16$, noise)' if b=='XS' else b
    ax1.plot(xs,absloss[b],'-o' if not dashed else '--o',color=colors[b],lw=1.6,ms=5,
             label=lbl,markerfacecolor=colors[b] if not dashed else 'white',markeredgecolor=colors[b])
    ax2.plot(xs,relloss[b],'-o' if not dashed else '--o',color=colors[b],lw=1.6,ms=5,
             markerfacecolor=colors[b] if not dashed else 'white',markeredgecolor=colors[b])
style(ax1); style(ax2)
ax1.set_ylabel(r'INT8 AP lost  ($\Delta$ mAP@[.5:.95], pts)')
ax2.set_ylabel("share of bin's own FP32 AP lost (%)")
ax1.set_title('absolute loss',fontsize=9,loc='left',color='0.3')
ax2.set_title('relative loss',fontsize=9,loc='left',color='0.3')
ax2.set_ylim(-8,35)   # clip: XS(n=16) relative loss is noise (11n spikes to -148%), keep S/M/L/XL readable
h,l=ax1.get_legend_handles_labels()
fig.legend(h,l,fontsize=7.5,loc='lower center',ncol=5,framealpha=0.9,
           title='sign height bin',title_fontsize=8,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=[0,0.08,1,1])
fig.savefig('figs/fig1_delta_fan.pdf',bbox_inches='tight')
fig.savefig('figs/fig1_delta_fan.png',dpi=150,bbox_inches='tight')
print("saved fig1 (5 points)")
for m,p in zip(order,xs): print(f"  {m}={p}M: dXL_pts={absloss['XL'][order.index(m)]:.1f}")
