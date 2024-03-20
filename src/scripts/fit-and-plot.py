import numpy as np
import matplotlib.pyplot as plt

import paths

x, y, yerr = np.loadtxt(paths.data / 'my-latest-greatest-data.txt')

X = np.array([np.ones(x.size), x]).T
W = np.linalg.inv(np.diag(np.square(yerr)))
beta = X.T @ W @ y @ np.linalg.inv(X.T @ W @ X)
intercept, gradient = beta

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(x, y, yerr, ls='', color='#5CDB94')
ax.plot(x, gradient * x + intercept, '-', 'k')
ax.set_ylabel('$y$')
ax.set_xlabel('$x$')
fig.savefig(paths.figures / 'my_plot.pdf')
plt.close()