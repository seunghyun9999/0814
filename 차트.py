import pandas as pd
import numpy
import matplotlib.pyplot as plt
import random

x=[random.uniform(0,100)for _ in range(1000)]
y=[random.uniform(0,100)for _ in range(1000)]
plt.clf()
fig, ax = plt.subplots()

plt.scatter(x,y, label='random',color='gold',marker='*',
            s=30, alpha=0.5)
ax.set_facecolor('black')
plt.xlabel('x')
plt.ylabel('y')
plt.show()