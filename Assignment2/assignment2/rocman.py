import matplotlib.pyplot as pl

pl.plot([0, 1])
pl.plot([0, .77, .58, .76, .62, 0.73,0.71, .73, 0.70, 0.70, 0.70, 0.71, 0.70, 0.73, 1], [0,.91, .87, .904, .89, 0.904, 0.89, 0.904, .89, .91, .90, .91, .90, .91, 1], 'r-')
pl.xlabel("False Positive Rate(FPR)")
pl.ylabel("True Positive Rate(TPR)")
pl.show()
