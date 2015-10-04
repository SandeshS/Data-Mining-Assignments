import matplotlib.pyplot as pl

pl.plot([0, 1])
pl.plot([0, .9, .85, .89, .86, 0.73,0.70, .71, 0.67, 0.68, 0.68, 0.71, 0.67, 0.68, 0.7, 0.79, 1], [0,.76, .55, .76, .64, 0.9, 0.88, 0.904, .89, .909, .904, .91, .904, .91, .92, .93, 1], 'r-')
pl.xlabel("False Positive Rate(FPR)")
pl.ylabel("True Positive Rate(TPR)")
pl.show()
