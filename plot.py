import mplfinance as mf
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.dates import num2date
import pandas as pd


BAR_WIDTH = 0.8

def plot_chart(data, action):
  dates = [num2date(d) for d in data[:,0]]
  ohlc_data = data[:,1:6]

  index = pd.DatetimeIndex(dates)
  ohlc = pd.DataFrame(data=ohlc_data, index=index,
                      columns=('Open', 'High', 'Low', 'Close', 'Volume'))

  max_y = ohlc['High'].max()
  min_y = ohlc['Low'].min()
  vls = [dates[60]]
  smas = []
  for c in (6, 7, 8, 9, 10):
    smas.append(
      [(d, v) for d, v in zip(dates, data[:, c])])

  # Candle stick chart.
  fig, axes = mf.plot(
    ohlc, volume=True, type='candlestick', style='yahoo', returnfig=True,
    # SMAs
    alines=dict(
      alines=smas,
      colors=['orange', 'turquoise', 'violet', 'lightgreen', 'royalblue'],
      linewidths=0.1,
      alpha=0.5),
    # Buy/Sell action.
    fill_between=dict(
      y1=max_y,
      y2=min_y,
      where=action['mask'],
      alpha=0.3,
      color='g' if action['type'] == 0 else 'r'),
    # Seprate obs from episode.
    vlines=dict(
      vlines=vls,
      linewidths=1,
      colors='gray',
      alpha=0.5))
  axes[2].set_ylabel('Volume')

  plt.show()
