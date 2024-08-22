import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import  scatter_matrix

header= ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv('./data/pima-indians-diabetes.data.csv',
                   names=header)
plt.clf()
#
data.plot(kind='box',subplots=True, figsize=(12,10),layout=(3,3),sharex=False)
plt.show()