from datetime import datetime
import glob
from matplotlib import pyplot
from matplotlib.dates import date2num
import numpy as np
import os
import pandas as pd


def scan():
  fs = glob.glob('data/train/*.npy')
  for f in fs:
    d = np.load(f)

    if np.any(d == 0.0):
      where = np.where(d == 0.0)
      # Volume col may be 0, that is ok.
      if np.any(where[1] != 5):
        print(f)
        print(where[0][where[1] != 5])
        print(where[1][where[1] != 5])


def plot():
  f = 'data/train/PCYG.npy'
  d = np.load(f)

  pyplot.plot(d[:,1])
  pyplot.plot(d[:,2])
  pyplot.plot(d[:,3])
  pyplot.plot(d[:,4])
  pyplot.show()


def plot_progress():
  p = pd.read_csv('__out__/atm_202010180723/progress.csv')
  p.plot(x='Step', y='Eval', style='-')
  pyplot.show()


if __name__ == '__main__':
  #scan()
  #plot()
  plot_progress()
