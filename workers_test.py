import agents
import configs
import unittest
import workers


# A mock ATM worker that acts according to a pre-defined script.
class MockAgent(agents.Agent):
  def __init__(self, actions):
    self._idx = 0
    self._actions = actions

  def get_action(self, obs):
    self._idx += 1
    return self._actions[self._idx - 1]


class WorkersTest(unittest.TestCase):
  def test_render_buy(self):
    # Init to hold.
    mock_actions = [1] * 30
    # Buy at the 8th day, and sell at the 20th day.
    mock_actions[8] = workers.T.BUY
    # Buy again. Should do nothing.
    mock_actions[10] = workers.T.BUY
    mock_actions[20] = workers.T.SELL
    worker = workers.ATM(configs.atm)
    worker.eval(MockAgent(mock_actions), render=True)

  def test_render_sell(self):
    # Init to hold.
    mock_actions = [1] * 30
    # First short.
    mock_actions[3] = workers.T.SELL
    mock_actions[10] = workers.T.BUY
    # Second long.
    mock_actions[18] = workers.T.BUY
    mock_actions[28] = workers.T.SELL
    worker = workers.ATM(configs.atm)
    worker.eval(MockAgent(mock_actions), render=True)


if __name__ == '__main__':
  unittest.main()
