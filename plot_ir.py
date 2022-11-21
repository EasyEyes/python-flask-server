import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/Users/ritikamotwani/easyeyes/python-flask-server/ir_py.csv',names=['one','two','three'])
plt.xscale('log')
[x,y] = plt.psd(df['one'],Fs=96000,scale_by_freq=False)
plt.minorticks_on()
plt.tick_params('both', length=10, width=2, which='minor')
plt.tick_params('both', length=20, width=3, which='major')

plt.show()
