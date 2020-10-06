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
    mock_actions[8] = 0
    mock_actions[20] = 2
    worker = workers.ATM(configs.atm)
    worker.eval(MockAgent(mock_actions), render=True)

  def test_render_sell(self):
    # Init to hold.
    mock_actions = [1] * 30
    # Sell at the 9th day, and buy at the 28th day.
    mock_actions[9] = 2
    mock_actions[28] = 0
    worker = workers.ATM(configs.atm)
    worker.eval(MockAgent(mock_actions), render=True)


if __name__ == '__main__':
  unittest.main()
