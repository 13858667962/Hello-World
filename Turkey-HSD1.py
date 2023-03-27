import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway

a = np.loadtxt('预实验.txt')
b = np.zeros(shape=(4, 6))
for i in range(6):
    for j in range(4):
        value = 0
        for k in range(25):
            value += a[j * 25 + k, i]
        value /= 25
        b[j, i] = value

df = pd.DataFrame(b)
print(df)
anova = f_oneway(b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 5])
print(anova)

fig, ax = plt.subplots(1, 1)
ax.boxplot([b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 5]])
ax.set_xticklabels(['1', '1.5', '2', '2.5', '3', '3.5'])
ax.set_ylabel('RPD')
plt.show()

res = stats.tukey_hsd(b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 5])
print(res)

hsd = 0.024
x_ticks = ('1', '1.5', '2', '2.5', '3', '3.5')
x = np.arange(1, 7)
y = np.mean(a, 0)
err = [hsd / 2 for _ in range(6)]
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.errorbar(x_ticks, y, yerr=err, capsize=10, linestyle='None',marker='o',)
plt.xlabel(chr(951),fontsize=20)
plt.ylabel('ARPD',fontsize=20)
plt.title('The means plot and confidence intervals at 95% confidence level',fontsize=20)
plt.show()
