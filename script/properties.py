import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from modlamp.descriptors import *
from utils import config, split_data
from scipy import stats
import itertools
plt.rcParams.update(config)

def create_multi_bars(tick_labels, datas, xlabel, tick_step=1, group_gap=0.2, bar_gap=0, labels=[], colors=[]):
    ticks = np.arange(len(tick_labels)) * tick_step
    group_num = len(datas)
    group_width = tick_step - group_gap
    bar_span = group_width / group_num
    bar_width = bar_span - bar_gap
    baseline_x = ticks - (group_width - bar_span) / 2
    f, ax = plt.subplots(figsize=(15,10))
    for index, y in enumerate(datas):
        plt.bar(baseline_x + index*bar_span, y, bar_width, label=labels[index], color=colors[index])
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.xticks(ticks, tick_labels)
    plt.legend(edgecolor='white', labelspacing=0.1, handletextpad=0.1,  handlelength=0.8,  prop=font)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #plt.savefig(f'Amino acids Frequency.svg', dpi=300, bbox_inches='tight')
    plt.show()

def get_aa_composition(seqs):
    STD_AA = list('ACDEFGHIKLMNPQRSTVWY')
    seq = ''.join(seqs)
    count = len(seq)

    aa_comp = {}
    for aa in STD_AA:
        aa_comp[aa] = seq.count(aa) / count
    return aa_comp

def aa_class(seqs)
    aa_comp =get_aa_composition(seqs)      
    aa_class={}
    n_p,n_ap,n_ar,n_pos,n_neg=0,0,0,0,0
    for aa in list('STCYNQDEKRH'):
        n_p+=aa_comp[aa]
    for  aa in list('GAVLIMFWPDEKRH'):
        n_ap+=aa_comp[aa]
    aa_class['apolar']=n_ap
    for aa in list('FWY'):
        n_ar+=aa_comp[aa]
    for aa in list('KRH'):
        n_pos+=aa_comp[aa]
    for aa in list('DE'):
        n_neg+=aa_comp[aa]
        
    aa_class['Apolar']=n_ap
    aa_class['Polar']=n_p 
    aa_class['Aromatic']=n_ar
    aa_class['Positive']=n_pos
    aa_class['Negative']=n_neg
    return aa_class
    
def add_significance_bars(ax, data, positions, pairs=None, method='mannwhitney'):
    if pairs is None:
        pairs = [(i, i+1) for i in range(len(data)-1)]

    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]

    valid_data = [d for d in data if len(d) > 0]
    if not valid_data: return 0

    base_h = max([max(d) for d in valid_data])    
    h_step = y_range * 0.20 
    text_offset = y_range * 0.05   
    top_y = base_h
    
    for i, (idx1, idx2) in enumerate(pairs):
        d1 = data[idx1]
        d2 = data[idx2] 
        if method == 'mannwhitney':
            stat, p_val = stats.mannwhitneyu(d1, d2, alternative='two-sided')
        elif method == 'ttest_ind':
            stat, p_val = stats.ttest_ind(d1, d2, equal_var=False)
        else:
            p_val = 1.0

        if p_val > 0.05: sig_symbol = 'ns'
        elif p_val > 0.01: sig_symbol = '*'
        elif p_val > 0.001: sig_symbol = '**'
        elif p_val > 0.0001: sig_symbol = '***'
        else: sig_symbol = '****'
            
        y_h = base_h + (i + 1) * h_step
        y_d = y_range * 0.03 
        x1, x2 = positions[idx1], positions[idx2]
        
        ax.plot([x1, x1, x2, x2], [y_h - y_d, y_h, y_h, y_h - y_d], lw=1.5, c='k')
        
        ax.text((x1 + x2) * 0.5, y_h + text_offset, sig_symbol, 
        ha='center', va='center', color='k', fontsize=20)
        top_y = y_h + text_offset
        
    return top_y
    
def create_multi_violin(datas, tick_labels, ylabel, ylim, fig_name, colors=[], pairs_to_compare=None, method='mannwhitney'):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if len(colors) < len(datas):
        colors = colors * (len(datas) // len(colors) + 1)

    positions = [i + 1 for i in range(len(datas))]

    for i, l in enumerate(datas):
        clean_data = [x for x in l if not np.isnan(x)]
        if not clean_data: continue
        vplot = ax.violinplot(clean_data, positions=[positions[i]], widths=0.5, showmeans=True, showmedians=False)

        vplot['cmeans'].set_edgecolor('black')
        vplot['cmeans'].set_linestyle('--')
        vplot['cmeans'].set_linewidth(2)
        vplot['cbars'].set_visible(False)
        vplot['cmins'].set_edgecolor('white')
        vplot['cmaxes'].set_edgecolor('white')
        for pc in vplot['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.8)
            pc.set_edgecolor(colors[i])
            pc.set_linewidth(0.8)
            
    if pairs_to_compare is None:
        pairs_to_compare = list(itertools.combinations(range(len(datas)), 2))

    highest_annotation = add_significance_bars(ax, datas, positions, pairs=pairs_to_compare, method=method)

    current_ylim = ax.get_ylim()
    y_range = current_ylim[1] - current_ylim[0]
    
    auto_top = highest_annotation + (y_range * 0.1) 
    
    if ylim:
        plt.ylim(ylim[0], max(ylim[1], auto_top))
    else:
        plt.ylim(bottom=current_ylim[0], top=auto_top)

    plt.xticks(positions, tick_labels, fontsize=31)
    plt.ylabel(ylabel, fontsize=37)
    plt.yticks(fontsize=31)
    plt.tick_params(axis='both', width=1.5)
    #plt.savefig(f'{fig_name}.svg', dpi=300, bbox_inches='tight')   
    plt.show()

def calc_charge(seqs, amide=True):
    g = GlobalDescriptor(seqs)
    g.calculate_charge(ph=7.4, amide=True)
    return list(g.descriptor[:,0])

def calc_isoelectricpoint(seqs, amide=True):
    g = GlobalDescriptor(seqs)
    g.isoelectric_point(amide=amide)
    calc = g.descriptor[:,0]
    return list(g.descriptor[:,0])

def calc_charge_density(seqs, amide=True):
    g = GlobalDescriptor(seqs)
    g.charge_density(ph=7.4, amide=amide)
    calc = g.descriptor[:,0]
    return list(g.descriptor[:,0])

def calc_instabilityindex(seqs):
    g = GlobalDescriptor(seqs)
    g.instability_index()
    calc = g.descriptor[:,0]
    return list(g.descriptor[:,0])

def calc_aromaticity(seqs):
    g = GlobalDescriptor(seqs)
    g.aromaticity()
    return list(g.descriptor[:,0])

def calc_aliphaticindex(seqs):
    g = GlobalDescriptor(seqs)
    g.aliphatic_index()
    return list(g.descriptor[:,0])    

def calc_bomanindex(seqs):
    g = GlobalDescriptor(seqs)
    return list(g.descriptor[:,0])

def calc_hydrophobicratio(seqs):
    g = GlobalDescriptor(seqs)
    g.hydrophobic_ratio()
    return list(g.descriptor[:,0])

def calc_hmoment(seqs):
    p= PeptideDescriptor(seqs, 'eisenberg')
    p.calculate_moment()
    return list(p.descriptor[:,0])

def calc_globalhydrophobicity(seqs):
    p =PeptideDescriptor(seqs,'eisenberg')
    p.calculate_global()
    return list(p.descriptor[:,0])

def calc_gravy(seqs):
    res=[]
    for seq in seqs:
        analysed_seq = ProteinAnalysis(seq)
        res.append(analysed_seq.gravy())
    return res

def calc_all(data, tick_labels, fig_name, colors=['#B1DCAF','#FFCCD1','#9BD2E6'], amide=True):
    
    res1, res2, res3, res4, res5, res6, res7, res8, res9, res10=[], [], [], [], [], [], [], [], [], []
    
    for i in range(len(data)):
        res3.append(calc_charge_density(data[i], amide=amide))
    create_multi_violin(res3, tick_labels, ylabel='Charge density', ylim=(-0.01,0.01), fig_name=f'{fig_name}_Charge_dendity', colors=colors)
    
    for i in range(len(data)):
        res1.append(calc_charge(data[i], amide=amide))
    create_multi_violin(res1, tick_labels, ylabel='Global charge', ylim=(-5,25), fig_name=f'{fig_name}_Global_charge', colors=colors)
    
    for i in range(len(data)):
        res8.append(calc_hmoment(data[i]))
    create_multi_violin(res8, tick_labels, ylabel='Hydrophobic moment', ylim=(), fig_name=f'{fig_name}_Hydrophobic moment', colors=colors)
    
    for i in range(len(data)):
        res7.append(calc_hydrophobicratio(data[i]))
    create_multi_violin(res7, tick_labels, ylabel='Hydrophobic ratio', ylim=(), fig_name=f'{fig_name}_Hydrophobic ratio', colors=colors)
    
    for i in range(len(data)):
        res9.append(calc_globalhydrophobicity(data[i]))
    create_multi_violin(res9, tick_labels, ylabel='Global hydrophobicity', ylim=(), fig_name=f'{fig_name}_Global_hydrophobicity', colors=colors)
    
    for i in range(len(data)):
        res2.append(calc_isoelectricpoint(data[i], amide=amide))
    create_multi_violin(res2, tick_labels, ylabel='Isoelectric point', ylim=(), fig_name=f'{fig_name}_Isoelectric_point', colors=colors)
    
    for i in range(len(data)):
        res6.append(calc_aliphaticindex(data[i]))
    create_multi_violin(res6, tick_labels, ylabel='Aliphatic index',ylim=(),  fig_name=f'{fig_name}_Aliphatic_index', colors=colors)     
    
    for i in range(len(data)):
        res5.append(calc_aromaticity(data[i]))
    create_multi_violin(res5, tick_labels, ylabel='Aromaticity', ylim=(), fig_name=f'{fig_name}_Aromaticity', colors=colors) 
    
    for i in range(len(data)):
        res4.append(calc_instabilityindex(data[i]))
    create_multi_violin(res4, tick_labels, ylabel='Instability index',ylim=(),  fig_name=f'{fig_name}_Instability_index', colors=colors)    
    
    for i in range(len(data)):
        res10.append(calc_gravy(data[i]))              
    create_multi_violin(res10, tick_labels, ylabel='Gravy', ylim=(), fig_name=f'{fig_name}_Gravy', colors=colors) 
    
if __name__ == '__main__':          
    AMP_df = pd.read_csv('../data/ecoli.csv')
    AMP_seqs=AMP_df['sequence'].tolist()   
    nonAMP_df=pd.read_csv('../data/nonAMP.csv')
    nonAMP_seqs=nonAMP_df['sequence'].tolist()
    #Amino acids Frequency
    AMPfraction=get_aa_composition(AMP_seqs)
    nonAMPfraction=get_aa_composition(nonAMP_seqs)
    create_multi_bars(AMPfraction.keys(), [list(AMPfraction.values()),list(nonAMPfraction.values()),], xlabel='Amino acids', labels=['AMP','nonAMP']) 
    #class Frequency               
    AMP_aaclass=aa_class(AMP_seqs)
    nonAMP_aaclass=aa_class(nonAMP_seqs)
    create_multi_bars(AMP_aaclass.keys(), [list(AMP_aaclass.values()),list(nonAMP_aaclass.values()),], xlabel='Amino acids classification', labels=['AMP','nonAMP']) 
    #modlamp plot volin
    data=[AMP_seqs,nonAMP_seqs]
    tick_labels=['AMP','nonAMP'] 
    calc_all(data,tick_labels, fig_name='AMP_nonAMP', amide=False)               