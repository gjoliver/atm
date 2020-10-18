import argparse
from datetime import datetime
import glob
from matplotlib.dates import date2num
import numpy as np
import os
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir',
                    type=str,
                    help='Dir of historical data files.')

parser.add_argument(
  '--format',
  type=str,
  help=('"old" has adj columns for all of open/high/low/close/volume, '
        'while "new" only has adj-close, and does NOT support splits.'))

parser.add_argument(
  '--price_scale',
  type=str,
  default='log',
  help=('Whether to use log scale or linear scale price.'))

args = parser.parse_args()


def date_to_float(date_str):
  return date2num(datetime.strptime(date_str, '%Y-%m-%d'))


def convert_old_file(df):
  # Turn string datetime into matplotlib float values.
  df.date = df.date.map(date_to_float)

  # Keep only the adjusted columns.
  df.drop(
    df.columns.difference(['date', 'adj_open', 'adj_high', 'adj_low',
                           'adj_close', 'adj_volume']),
    1, inplace=True)

  df.rename({
    'date': 'date',
    'adj_open': 'open',
    'adj_high': 'high',
    'adj_low': 'low',
    'adj_close': 'close',
    'adj_volume': 'volume',
  }, axis=1, inplace=True)

  return df


def convert_new_file(df):
  # Turn string datetime into matplotlib float values.
  df.Date = df.Date.map(date_to_float)

  offset = df['Adj Close'] - df['Close']
  df.drop(
    df.columns.difference(
      ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']),
    1, inplace=True)

  # Offset all of the price columns based on the diffs between
  # Close and Adj Close.
  # Note(jungong) : this makes sense because we don't support splits.
  for c in ['Open', 'High', 'Low', 'Close']:
    df[c] = df[c] + offset

  df.rename({
    'Date': 'date',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume',
  }, axis=1, inplace=True)

  return df


# Columns of output data.
# Date, Open, High, Low, Close, Volumn, 8-Day, 20-Day, 50-Day, 100-Day, 200-Day
def convert_file(path):
  df = pd.read_csv(path, sep=',', header=0)

  # Drop rows with any missing data.
  df = df.dropna(how = 'any')

  if args.format == 'old':
    df = convert_old_file(df)
  elif args.format == 'new':
    df = convert_new_file(df)
  else:
    assert False, 'Must specify --format flag.'

  # Compute 8, 20, 50, 100, 200, SMA.
  for ma in [8, 20, 50, 100, 200]:
    col_name = 'sma{}'.format(ma)
    df[col_name] = df.iloc[:,3].rolling(window=ma, min_periods=1).mean()

  # Turn all of the price columns into log scale,
  # so sudden pop/drop doesn't screw our percentage diff computation.
  # TODO(jungong) : maybe we should do this before calculating all the
  # MAs. Not sure.
  if args.price_scale == 'log':
    for col in ('open',
                'high',
                'low',
                'close',
                'sma8',
                'sma20',
                'sma50',
                'sma100',
                'sma200'):
      df[col] = df[col].apply(lambda x: np.log(x + 1))

  return df


def convert():
  fs = glob.glob(os.path.join(args.input_dir, '*.csv'))
  for f in fs:
    print('Converting ... {}'.format(f))

    d = convert_file(f).to_numpy()

    # We know f ends with '.csv'.
    np.save(f[:-4], d)


if __name__ == '__main__':
  convert()
