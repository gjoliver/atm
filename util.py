import math
import numpy as np


def ScalePL(price, entry_price):
  pct = (price - entry_price) / entry_price
  # e^0.69314718 = 2.0
  # 0.05 / 0.69314718 = 0.0721347521
  # So the following formula will re-scale [-1.0, 1.0] to [-1.0, 1.0],
  # with 0.05 (5% gain or loss) -> +/-0.5.
  return np.sign(pct) * (1.0 - 1.0 / math.exp(abs(pct) / 0.0721347521))


def ScaleLinear(x, min, max):
  if min == max: return 0
  return (x - min) / (max - min)
