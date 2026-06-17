import random
from cg.api import Observation, search_begin, search_step
from src.base_agent import BaseAgent
from .ga import DEFAULT_WEIGHTS, evaluate_state

class EvolutionaryAgent(BaseAgent):
    """
    シミュレータを用いて 1 ステップ先のアクション結果を予測し、
    盤面評価関数のスコアが最大となるアクションを選択する進化計算エージェント。
    """
    def __init__(self, deck_path="deck.csv", weights=None):
        super().__init__(deck_path)
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS
        
    def _select_action_impl(self, obs: Observation) -> list[int]:
        try:
            # 1. 探索シミュレーション用の予測カード情報の構築
            my_deck = self._predict_my_deck(obs)
            my_prize = self._predict_my_prize(obs)
            
            # 相手の情報を予測（仮で自分と同じデッキ構成とする）
            opp_deck = self.read_deck_csv()
            opp_prize = [1] * len(obs.current.players[1 - obs.current.yourIndex].prize)
            opp_hand = [1] * obs.current.players[1 - obs.current.yourIndex].handCount
            
            # 相手のバトル場が裏向きの場合は仮のID（タケルライコex: 63）を設定
            opp_active = []
            opp_active_state = obs.current.players[1 - obs.current.yourIndex].active
            if len(opp_active_state) > 0 and opp_active_state[0] is None:
                opp_active = [63]
            
            # 2. 探索木のルート状態を初期化
            root_state = search_begin(
                agent_observation=obs,
                your_deck=my_deck,
                your_prize=my_prize,
                opponent_deck=opp_deck,
                opponent_prize=opp_prize,
                opponent_hand=opp_hand,
                opponent_active=opp_active
            )
            
            # 3. 1ステップ先の予測と評価関数の適用
            best_score = -float('inf')
            best_action = None
            
            min_c = obs.select.minCount
            max_c = obs.select.maxCount
            option_count = len(obs.select.option)
            
            # 評価対象のアクション組み合わせ候補を作成します
            candidates = []
            if max_c == 1:
                # 選択肢を1つ選ぶ場合
                candidates = [[i] for i in range(option_count)]
            else:
                # 複数選択肢を選ぶ場合、ランダムにいくつかの組み合わせをサンプリングします
                # (実行時間制限を遵守するため、最大50候補に制限)
                for _ in range(50):
                    # minCount〜maxCountの間で選択数を決定
                    count = random.randint(min_c, max_c)
                    cand = random.sample(range(option_count), count)
                    cand.sort()
                    if cand not in candidates:
                        candidates.append(cand)
            
            # 各候補アクションを実行した後の状態を評価
            for cand in candidates:
                try:
                    # アクションを実行して次の状態を予測
                    next_state = search_step(root_state.searchId, cand)
                    next_obs = next_state.observation
                    
                    # 状態評価スコアの計算
                    score = evaluate_state(next_obs, self.weights)
                    if score > best_score:
                        best_score = score
                        best_action = cand
                except Exception:
                    # シミュレータ側での一部アクション予測エラーはスキップ
                    continue
            
            if best_action is not None:
                return best_action
                
        except Exception as e:
            # 探索のセットアップ自体が失敗した場合はログを出力してランダムにフォールバック
            print(f"探索シミュレーションに失敗したため、ランダム選択にフォールバックします: {e}")
            
        # フォールバック: ランダムに選択
        return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)

    def _predict_my_deck(self, obs: Observation) -> list[int]:
        """
        公開情報（トラッシュ、手札、場）から自分の山札に残っているカードIDを逆算します。
        """
        deck = self.read_deck_csv()
        your_idx = obs.current.yourIndex
        me = obs.current.players[your_idx]
        
        # トラッシュにあるカードを除外
        for card in me.discard:
            if card.id in deck:
                deck.remove(card.id)
                
        # 手札にあるカードを除外
        if me.hand is not None:
            for card in me.hand:
                if card.id in deck:
                    deck.remove(card.id)
                    
        # 場のポケモンおよび付随カードを除外
        def remove_pokemon_cards(p):
            if p is not None:
                if p.id in deck:
                    deck.remove(p.id)
                for card in p.energyCards:
                    if card.id in deck:
                        deck.remove(card.id)
                for card in p.tools:
                    if card.id in deck:
                        deck.remove(card.id)
                for card in p.preEvolution:
                    if card.id in deck:
                        deck.remove(card.id)
                        
        for p in me.active:
            remove_pokemon_cards(p)
        for p in me.bench:
            remove_pokemon_cards(p)
            
        # 残り枚数を Observation に記載されている山札枚数に調整
        while len(deck) > me.deckCount and len(deck) > 0:
            deck.pop()
        while len(deck) < me.deckCount:
            deck.append(1)  # 不足分は基本草エネルギーで補填
            
        return deck

    def _predict_my_prize(self, obs: Observation) -> list[int]:
        """
        自分のサイドカードの枚数に合わせて、ダミーのカードID（基本草エネルギー: 1）を割り当てます。
        """
        your_idx = obs.current.yourIndex
        prize_count = len(obs.current.players[your_idx].prize)
        return [1] * prize_count
