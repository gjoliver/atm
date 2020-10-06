import mplfinance as mf
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.dates import num2date
import pandas as pd
import workers


BAR_WIDTH = 0.8

def _get_bitmask_and_colors(mask):
  bitmask = []
  colors = []
  in_fill = False
  for m in mask:
    if m == workers.PT.NO_POSITION:
      bitmask.append(False)
      if in_fill: in_fill = False
      continue

    bitmask.append(True)
    if not in_fill:
      # Start of a new trade.
      in_fill = True
      colors.append('g' if m == workers.PT.LONG else 'r')

  return bitmask, colors


def plot_chart(data, mask):
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

  fill_mask, fill_colors = _get_bitmask_and_colors(mask)

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
      where=fill_mask,
      alpha=0.3,
      color=fill_colors),
    # Seprate obs from episode.
    vlines=dict(
      vlines=vls,
      linewidths=1,
      colors='gray',
      alpha=0.5))
  axes[2].set_ylabel('Volume')

  plt.show()
