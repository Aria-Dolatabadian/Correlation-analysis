#install pingouin in anaconda
#pip install pingouin
#or
#pip install --upgrade pingouin
# import data
import pandas as pd
df = pd.read_csv (r'corr.csv')
print (df)
# Simple correlation between two columns
import pingouin as pg
pg.corr(x=df['Height'], y=df['Weight'])
print(pg.corr(x=df['Height'], y=df['Weight']))
# How does the correlation look visually?
import seaborn as sns
import matplotlib.pyplot as plt
#from scipy.stats import pearsonr
sns.set(style='white', font_scale=1.2)
g = sns.JointGrid(data=df, x='Height', y='Weight', xlim=(140, 190), ylim=(40, 100), height=5)
g = g.plot_joint(sns.regplot, color="xkcd:muted blue")
g = g.plot_marginals(sns.distplot, kde=False, bins=12, color="xkcd:bluey grey")
g.ax_joint.text(145, 95, 'r = 0.45, p < .001', fontstyle='italic')
plt.tight_layout()
plt.show()

pg.pairwise_corr(df).sort_values(by=['p-unc'])[['X', 'Y', 'n', 'r', 'p-unc']]
print(pg.pairwise_corr(df).sort_values(by=['p-unc'])[['X', 'Y', 'n', 'r', 'p-unc']])

df.corr().round(2)
print(df.corr().round(2))
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
corrs = df.corr()
mask = np.zeros_like(corrs)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corrs, cmap='Spectral_r', mask=mask, square=True, vmin=-.4, vmax=.4)
plt.title('Correlation matrix')
plt.show()

df.rcorr(stars=False)
print(df.rcorr(stars=False))

df[['O', 'C', 'E', 'A', 'N']].rcorr()
print(df[['O', 'C', 'E', 'A', 'N']].rcorr())
