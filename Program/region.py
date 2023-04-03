#!/c/Users/Bardi/AppData/Local/Programs/Python/Python39/python -i
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
gdp = pd.read_csv("GDP.csv")
pref_mono = pd.read_csv("monohulled_rpref.csv")
# pref_mono.drop('USNowhere', axis=1)
# pref_mono.drop('EUNowhere', axis=1)

hkgdp = 49000

pref_mono['gdp']=pref_mono['CRS'].map(gdp.set_index('CRS').squeeze())
pref_mono = pref_mono.groupby(pd.cut(pref_mono['gdp'], bins=8)).mean()
pref_mono = pref_mono.dropna()
pref_cata = pd.read_csv("catamarans_rpref.csv")
pref_cata['gdp']=pref_cata['CRS'].map(gdp.set_index('CRS').squeeze())
pref_cata = pref_cata.groupby(pd.cut(pref_cata['gdp'], bins=8)).mean()
pref_cata = pref_cata.dropna()

print("--mono\nCorr (pearson): {:.3f}".format(pref_mono['Price'].corr(pref_mono['gdp'], method='pearson')))
print("Corr (spearman): {:.3f}".format(pref_mono['Price'].corr(pref_mono['gdp'], method='spearman')))
print("Corr (kendall): {:.3f}".format(pref_mono['Price'].corr(pref_mono['gdp'], method='kendall')))
_, p_mono = pearsonr(pref_mono['gdp'], pref_mono['Price'])
print("p: {:.5f}, significant".format(p_mono))

poly_mono = np.polyfit(pref_mono.gdp, pref_mono.Price, 1)
print(poly_mono)

print("--cata\nCorr (pearson): {:.3f}".format(pref_cata['Price'].corr(pref_cata['gdp'], method='pearson')))
print("Corr (spearman): {:.3f}".format(pref_cata['Price'].corr(pref_cata['gdp'], method='spearman')))
print("Corr (kendall): {:.3f}".format(pref_cata['Price'].corr(pref_cata['gdp'], method='kendall')))
_, p_cata = pearsonr(pref_cata['gdp'], pref_cata['Price'])
print("p: {:.5f}, significant".format(p_cata))

poly_cata = np.polyfit(pref_cata.gdp, pref_cata.Price, 1)
print(poly_cata)

print("For hk(mono) {:.3f}: (cata): {:.3f}".format(np.polyval(poly_mono, [hkgdp])[0], np.polyval(poly_cata, [hkgdp])[0]))

plt.xlabel('GDP per capita')
plt.ylabel('Weighted Average Variant Price')

pref_mono.set_index('gdp')['Price'].sort_index().plot(label = 'Monohulled Boats')
pref_cata.set_index('gdp')['Price'].sort_index().plot(label = 'Catamarans')
plt.plot(pref_mono.gdp, np.polyval(poly_mono, pref_mono.gdp), label="Monohulled, linear fitted")
plt.plot(pref_cata.gdp, np.polyval(poly_cata, pref_cata.gdp), label="Catamarans, linear fitted")

plt.legend()
plt.show()