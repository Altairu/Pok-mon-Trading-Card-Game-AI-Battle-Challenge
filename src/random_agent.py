import random
from cg.api import Observation
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    ランダムにアクションを選択するエージェント（ベースライン実装）。
    """
    def _select_action_impl(self, obs: Observation) -> list[int]:
        # 選択肢の中からランダムに maxCount 個選んで返します。
        return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)
