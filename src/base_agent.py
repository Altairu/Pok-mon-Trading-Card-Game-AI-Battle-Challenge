import os
from cg.api import Observation, to_observation_class

class BaseAgent:
    """
    すべてのエージェントの基底クラス。
    デッキの読み込みや初期選択処理など、共通処理を提供します。
    """
    def __init__(self, deck_path="deck.csv"):
        self.deck_path = deck_path
        
    def read_deck_csv(self) -> list[int]:
        """deck.csv から60枚のカードIDリストを読み込みます。"""
        file_path = self.deck_path
        if not os.path.exists(file_path):
            file_path = "/kaggle_simulations/agent/" + file_path
        with open(file_path, "r") as file:
            csv = file.read().split("\n")
        deck = []
        for i in range(60):
            deck.append(int(csv[i]))
        return deck
        
    def select_action(self, obs_dict: dict) -> list[int]:
        """
        Kaggle 環境から呼び出されるメインの選択関数。
        初期ターンはデッキを返し、通常ターンは各エージェントのロジックを実行します。
        """
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            # 初期選択時には60枚のデッキリストを返します。
            return self.read_deck_csv()
        
        return self._select_action_impl(obs)
        
    def _select_action_impl(self, obs: Observation) -> list[int]:
        """エージェント個別の意思決定ロジックを実装する抽象メソッド。"""
        raise NotImplementedError("サブクラスで _select_action_impl を実装してください。")
