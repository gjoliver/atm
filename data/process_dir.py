import argparse
import glob
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

args = parser.parse_args()


# Columns of output data.
# Open, High, Low, Close, Volumn, 8-Day, 20-Day, 50-Day, 100-Day, 200-Day
def convert_file(path):
  df = pd.read_csv(path, sep=',', header=0)

  if args.format == 'old':
    # Keep only the adjusted columns.
    df.drop(
      df.columns.difference(
        ['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']),
      1, inplace=True)
  elif args.format == 'new':
    offset = df['Adj Close'] - df['Close']
    df.drop(
      df.columns.difference(['Open', 'High', 'Low', 'Close', 'Volume']),
      1, inplace=True)
    # Offset all of the price columns based on the diffs between
    # Close and Adj Close.
    # Note(jungong) : this makes sense because we don't support splits.
    for c in ['Open', 'High', 'Low', 'Close']:
      df[c] = df[c] + offset
  else:
    assert False, 'Must specify --format flag.'

  # Compute 8, 20, 50, 100, 200, SMA.
  for ma in [8, 20, 50, 100, 200]:
    col_name = 'sma{}'.format(ma)
    df[col_name] = df.iloc[:,3].rolling(window=ma, min_periods=1).mean()

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
