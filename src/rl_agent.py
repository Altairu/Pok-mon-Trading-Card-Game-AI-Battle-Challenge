import random
import os
from cg.api import Observation, search_begin, search_step, search_end
from src.base_agent import BaseAgent
from .evolutionary.ga import DEFAULT_WEIGHTS, get_features, evaluate_state

class RlAgent(BaseAgent):
    """
    時間的差分学習（TD学習）に基づいて価値関数の重みパラメータを
    自己対戦シミュレーションやプレイを通じて学習・最適化する強化学習エージェント。
    """
    def __init__(self, deck_path="deck.csv", weights=None, learning_rate=0.0005, discount_factor=0.95, epsilon=0.1):
        super().__init__(deck_path)
        self.weights = weights if weights is not None else list(DEFAULT_WEIGHTS)
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon  # 探索確率 (ε-greedy)
        
        # 保存された最良の重みがあればロード
        self._load_saved_weights()

    def _load_saved_weights(self):
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolutionary", "best_weights.txt")
        if os.path.exists(weights_path):
            try:
                with open(weights_path, "r") as f:
                    vals = [float(x.strip()) for x in f.read().split(",") if x.strip()]
                    if len(vals) == len(self.weights):
                        import math
                        if not any(math.isnan(x) for x in vals):
                            self.weights = vals
                            print("RL用に最適化された重みパラメータをロードしました。")
                        else:
                            print("警告: 保存された重みにnanが含まれているため、ロードをスキップしました。")
            except Exception as e:
                print(f"重みパラメータのロードに失敗しました: {e}")

    def save_weights(self):
        """現在の学習された重みをファイルに保存します。"""
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolutionary", "best_weights.txt")
        try:
            with open(weights_path, "w") as f:
                f.write(",".join(map(str, self.weights)))
            print("RLの学習済み重みを保存しました。")
        except Exception as e:
            print(f"重みの保存に失敗しました: {e}")

    def _select_action_impl(self, obs: Observation) -> list[int]:
        try:
            # 決定論化（非公開情報の予測）
            my_deck = self._predict_my_deck(obs)
            my_prize = self._predict_my_prize(obs)
            opp_deck = self.read_deck_csv()
            opp_prize = [1] * len(obs.current.players[1 - obs.current.yourIndex].prize)
            opp_hand = [1] * obs.current.players[1 - obs.current.yourIndex].handCount
            opp_active = []
            opp_active_state = obs.current.players[1 - obs.current.yourIndex].active
            if len(opp_active_state) > 0 and opp_active_state[0] is None:
                opp_active = [63]

            root_state = search_begin(
                agent_observation=obs,
                your_deck=my_deck,
                your_prize=my_prize,
                opponent_deck=opp_deck,
                opponent_prize=opp_prize,
                opponent_hand=opp_hand,
                opponent_active=opp_active
            )
            
            option_count = len(obs.select.option)
            min_c = obs.select.minCount
            max_c = obs.select.maxCount
            
            # 1. アクション候補の作成
            candidates = []
            if max_c == 1:
                candidates = [[i] for i in range(option_count)]
            else:
                for _ in range(30):
                    count = random.randint(min_c, max_c)
                    cand = random.sample(range(option_count), count)
                    cand.sort()
                    if cand not in candidates:
                        candidates.append(cand)
            
            # ε-greedy による意思決定
            chosen_action = None
            if random.random() < self.epsilon:
                # 探索 (Exploration)
                chosen_action = random.choice(candidates)
            else:
                # 活用 (Exploitation) - 最も価値の高いアクションを選択
                best_score = -float('inf')
                for cand in candidates:
                    try:
                        next_state = search_step(root_state.searchId, cand)
                        score = evaluate_state(next_state.observation, self.weights)
                        if score > best_score:
                            best_score = score
                            chosen_action = cand
                    except:
                        continue
            
            if chosen_action is None:
                chosen_action = candidates[0]
            
            # --- TD学習によるパラメータの更新 (オンライン更新) ---
            # 選択したアクションを実行した後の価値を計算し、現在の重みを更新します
            try:
                next_state = search_step(root_state.searchId, chosen_action)
                next_obs = next_state.observation
                
                # 特徴量の取得と価値の計算
                features = get_features(obs)
                next_features = get_features(next_obs)
                
                import math
                v_s = sum(w * f for w, f in zip(self.weights, features))
                v_s_prime = sum(w * f for w, f in zip(self.weights, next_features))
                
                if not math.isnan(v_s) and not math.isinf(v_s) and not math.isnan(v_s_prime) and not math.isinf(v_s_prime):
                    # 報酬の定義 (サイドカードの獲得枚数に基づく即時報酬)
                    prize_before = len(obs.current.players[obs.current.yourIndex].prize)
                    prize_after = len(next_obs.current.players[obs.current.yourIndex].prize)
                    opp_prize_before = len(obs.current.players[1 - obs.current.yourIndex].prize)
                    opp_prize_after = len(next_obs.current.players[1 - obs.current.yourIndex].prize)
                    
                    reward = (prize_before - prize_after) * 2.0 - (opp_prize_before - opp_prize_after) * 2.0
                    
                    # 試合終了時の追加報酬
                    if next_obs.current.result == 0:
                        reward += 10.0 # 勝利
                    elif next_obs.current.result == 1:
                        reward -= 10.0 # 敗北
                    
                    # TD誤差の計算: delta = reward + gamma * V(s') - V(s)
                    td_error = reward + self.gamma * v_s_prime - v_s
                    
                    # TD誤差をクリッピングして勾配爆発を防ぐ
                    td_error = max(-50.0, min(50.0, td_error))
                    
                    # 重みの更新とクリッピング
                    new_weights = list(self.weights)
                    for i in range(len(self.weights)):
                        update = self.alpha * td_error * features[i]
                        # 1回の更新幅をクリップして緩やかに学習させる
                        update = max(-2.0, min(2.0, update))
                        if not math.isnan(update) and not math.isinf(update):
                            new_weights[i] += update
                            # 重みの値を一定範囲に制限
                            new_weights[i] = max(-500.0, min(500.0, new_weights[i]))
                    
                    # 更新後の重みに異常が無いかチェック
                    if not any(math.isnan(w) or math.isinf(w) for w in new_weights):
                        self.weights = new_weights
                        
                        # 定期的な重みの保存
                        if random.random() < 0.1:
                            self.save_weights()
            except Exception as e:
                # 更新時のエラーは無視して進行
                pass
                
            search_end()
            return chosen_action
            
        except Exception as e:
            print(f"RLエージェントの処理でエラーが発生したため、ランダム選択にフォールバックします: {e}")
            try:
                search_end()
            except:
                pass
            return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)

    def _predict_my_deck(self, obs: Observation) -> list[int]:
        deck = self.read_deck_csv()
        your_idx = obs.current.yourIndex
        me = obs.current.players[your_idx]
        for card in me.discard:
            if card.id in deck:
                deck.remove(card.id)
        if me.hand is not None:
            for card in me.hand:
                if card.id in deck:
                    deck.remove(card.id)
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
        while len(deck) > me.deckCount and len(deck) > 0:
            deck.pop()
        while len(deck) < me.deckCount:
            deck.append(1)
        return deck

    def _predict_my_prize(self, obs: Observation) -> list[int]:
        your_idx = obs.current.yourIndex
        prize_count = len(obs.current.players[your_idx].prize)
        return [1] * prize_count
