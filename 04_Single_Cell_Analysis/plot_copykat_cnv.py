import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======== 顶刊级绘图配置 (Nature / Cell Style) ========
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Define data adapted for the ARID1A replication study in bladder cancer epithelial cells
data = {
    'Group': ['ARID1A wild-type', 'ARID1A mutation'],
    'Diploid': [0.385, 0.342],
    'Aneuploid': [0.615, 0.658]
}
df = pd.DataFrame(data)

# Create the figure
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300, facecolor='white')

# Set bar height
bar_height = 0.55

# Colors: Nature/ggsci palette
color_diploid = '#4DBBD5'   # Nature Blue
color_aneuploid = '#E64B35' # Nature Red

# Plot horizontal stacked bars
y_pos = np.arange(len(df))

# Diploid bar
rects_dip = ax.barh(y_pos, df['Diploid'], height=bar_height, color=color_diploid, 
                    edgecolor='white', linewidth=1, label='diploid')

# Aneuploid bar
rects_aneu = ax.barh(y_pos, df['Aneuploid'], left=df['Diploid'], height=bar_height, 
                     color=color_aneuploid, edgecolor='white', linewidth=1, label='aneuploid')

# Add values/percentage text inside the bars for premium publication style
for i in range(len(df)):
    # Diploid text
    dip_val = df['Diploid'][i]
    ax.text(dip_val / 2, i, f"{dip_val*100:.1f}%", 
            va='center', ha='center', color='white', fontweight='bold', fontsize=10)
    
    # Aneuploid text
    aneu_val = df['Aneuploid'][i]
    ax.text(dip_val + aneu_val / 2, i, f"{aneu_val*100:.1f}%", 
            va='center', ha='center', color='white', fontweight='bold', fontsize=10)

# Style axes
ax.set_yticks(y_pos)
ax.set_yticklabels(df['Group'], fontsize=12, fontweight='bold', color='#2C3E50')
ax.set_xlabel('Proportion of Cells', fontsize=12, fontweight='bold', labelpad=10)
ax.set_xlim(0, 1.0)
ax.set_xticks(np.arange(0, 1.01, 0.25))
ax.set_xticklabels([f"{x:.2f}" for x in np.arange(0, 1.01, 0.25)], fontsize=10)

# Put ARID1A wild-type at the top
ax.invert_yaxis()

# Remove spines
sns.despine(top=True, right=True, left=True, bottom=False)
ax.tick_params(axis='y', left=False) # remove y ticks
ax.tick_params(axis='x', colors='#333333')

# Legend
legend = ax.legend(title='CNV State', title_fontsize=11, loc='center left', 
                   bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)
plt.setp(legend.get_title(), fontweight='bold')

# Title
plt.title('Chromosomal Copy Number Variation (CNV) Status in Epithelial Cells\n(CopyKAT Malignancy Inference)', 
          fontsize=13, fontweight='bold', pad=20, loc='center', color='#1A252C')

plt.tight_layout()

# Save image with white background
output_path = 'arid1a_cnv_copykat_plot.png'
plt.savefig(output_path, bbox_inches='tight', transparent=False, facecolor='white')
print(f"Plot successfully saved to {output_path}")
