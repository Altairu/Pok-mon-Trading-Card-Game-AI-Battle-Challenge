import random
import math
import os
from cg.api import Observation, search_begin, search_step, search_release, search_end
from src.base_agent import BaseAgent
from .evolutionary.ga import DEFAULT_WEIGHTS, evaluate_state

class MctsNode:
    """
    モンテカルロ木探索の木構造を表すノードクラス。
    """
    def __init__(self, search_id, obs, parent=None, action_from_parent=None):
        self.search_id = search_id
        self.obs = obs
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = []
        self.visit_count = 0
        self.total_value = 0.0
        
        # 実行可能な未探索アクションのリストを作成
        self.untried_actions = self._get_possible_actions()
        
    def _get_possible_actions(self):
        if not self.obs or not self.obs.select:
            return []
        
        min_c = self.obs.select.minCount
        max_c = self.obs.select.maxCount
        option_count = len(self.obs.select.option)
        
        actions = []
        if max_c == 1:
            actions = [[i] for i in range(option_count)]
        else:
            # 複数選択肢の場合はランダムサンプリングで最大15個に制限
            # 探索時間を一定に抑えるためのバウンディング
            import itertools
            if option_count <= 5:
                for r in range(min_c, max_c + 1):
                    for comb in itertools.combinations(range(option_count), r):
                        actions.append(list(comb))
            else:
                for _ in range(15):
                    count = random.randint(min_c, max_c)
                    cand = random.sample(range(option_count), count)
                    cand.sort()
                    if cand not in actions:
                        actions.append(cand)
        return actions

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        if not self.obs or not self.obs.current:
            return True
        return self.obs.current.result != -1

    def best_child(self, c_param=1.414):
        # 訪問回数が0のものがあれば優先選択
        for child in self.children:
            if child.visit_count == 0:
                return child
                
        best_score = -float('inf')
        best_node = None
        
        # 決定を下すプレイヤーが自分（0）か相手（1）か
        is_my_turn = (self.obs.current.yourIndex == 0)
        
        for child in self.children:
            avg_val = child.total_value / child.visit_count
            # UCB1のバイアス項
            exploration = c_param * math.sqrt(math.log(self.visit_count) / child.visit_count)
            
            # 相手のターンであれば、自分にとって価値が最小になる選択（相手の最善手）を想定する
            score = avg_val + exploration if is_my_turn else -avg_val + exploration
            
            if score > best_score:
                best_score = score
                best_node = child
                
        return best_node

class MctsAgent(BaseAgent):
    """
    モンテカルロ木探索（MCTS）と進化計算で得た最良の評価関数重みを組み合わせた
    ハイブリッド対戦エージェント。
    """
    def __init__(self, deck_path="deck.csv", weights=None, max_iterations=50, max_depth=3):
        super().__init__(deck_path)
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        
        # 保存された最良の重みがあればロード
        self._load_saved_weights()

    def _load_saved_weights(self):
        # best_weights.txt があればロードします
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolutionary", "best_weights.txt")
        if os.path.exists(weights_path):
            try:
                with open(weights_path, "r") as f:
                    vals = [float(x.strip()) for x in f.read().split(",") if x.strip()]
                    if len(vals) == 10:
                        self.weights = vals
                        print("MCTS用に最適化された重みパラメータをロードしました。")
            except Exception as e:
                print(f"重みパラメータのロードに失敗しました: {e}")

    def _select_action_impl(self, obs: Observation) -> list[int]:
        try:
            # 1. 決定論化（非公開領域のカードID予測）
            my_deck = self._predict_my_deck(obs)
            my_prize = self._predict_my_prize(obs)
            opp_deck = self.read_deck_csv()
            opp_prize = [1] * len(obs.current.players[1 - obs.current.yourIndex].prize)
            opp_hand = [1] * obs.current.players[1 - obs.current.yourIndex].handCount
            
            opp_active = []
            opp_active_state = obs.current.players[1 - obs.current.yourIndex].active
            if len(opp_active_state) > 0 and opp_active_state[0] is None:
                opp_active = [63] # ダミーのたねポケモンIDを設定

            # 2. 探索の開始
            root_state = search_begin(
                agent_observation=obs,
                your_deck=my_deck,
                your_prize=my_prize,
                opponent_deck=opp_deck,
                opponent_prize=opp_prize,
                opponent_hand=opp_hand,
                opponent_active=opp_active
            )
            
            root_node = MctsNode(root_state.searchId, obs)
            
            # 指定されたイテレーション回数分MCTSを実行
            for _ in range(self.max_iterations):
                node = root_node
                path_to_release = [] # プレイアウトなどで生成される一時ノードの解放管理用
                
                # --- SELECT ---
                while not node.is_terminal() and node.is_fully_expanded() and len(node.children) > 0:
                    next_node = node.best_child()
                    if next_node is None:
                        break
                    node = next_node
                
                # --- EXPAND ---
                if not node.is_terminal() and not node.is_fully_expanded():
                    action = node.untried_actions.pop()
                    try:
                        next_state = search_step(node.search_id, action)
                        child_node = MctsNode(next_state.searchId, next_state.observation, parent=node, action_from_parent=action)
                        node.children.append(child_node)
                        node = child_node
                    except Exception:
                        continue
                
                # --- SIMULATE (Playout) ---
                val = self._playout(node, path_to_release)
                
                # プレイアウト中の一時シミュレータ状態を速やかに解放
                for temp_id in path_to_release:
                    try:
                        search_release(temp_id)
                    except:
                        pass
                
                # --- BACKPROPAGATE ---
                while node is not None:
                    node.visit_count += 1
                    node.total_value += val
                    node = node.parent
            
            # 最も訪問回数の多いアクションを決定
            best_action = None
            if root_node.children:
                best_child = max(root_node.children, key=lambda c: c.visit_count)
                best_action = best_child.action_from_parent

            # 探索の終了とC++側リソースの一括解放
            search_end()
            
            if best_action is not None:
                return best_action
                
        except Exception as e:
            print(f"MCTS探索中にエラーが発生したため、ランダム選択にフォールバックします: {e}")
            try:
                search_end()
            except:
                pass

        # フォールバック
        return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)

    def _playout(self, node, path_to_release) -> float:
        """
        簡易評価関数またはランダムプレイングを用いて一定の深さまでシミュレーションし、
        到達した盤面の評価値を返します。
        """
        if node.is_terminal():
            # すでに勝負が決まっている場合
            res = node.obs.current.result
            if res == 0: # 自分(P1)の勝利
                return 10000.0
            elif res == 1: # 相手(P2)の勝利
                return -10000.0
            return 0.0

        current_id = node.search_id
        current_obs = node.obs
        
        # 最大深度まで仮想シミュレーションを回す
        for _ in range(self.max_depth):
            if not current_obs or not current_obs.select or current_obs.current.result != -1:
                break
                
            # ランダムにプレイアクションを選択
            min_c = current_obs.select.minCount
            max_c = current_obs.select.maxCount
            option_count = len(current_obs.select.option)
            
            count = random.randint(min_c, max_c)
            action = random.sample(range(option_count), count)
            action.sort()
            
            try:
                next_state = search_step(current_id, action)
                current_id = next_state.searchId
                current_obs = next_state.observation
                path_to_release.append(current_id)
            except Exception:
                break
                
        # 到達した最終状態を評価関数で評価
        if current_obs:
            if current_obs.current.result == 0:
                return 10000.0
            elif current_obs.current.result == 1:
                return -10000.0
            return evaluate_state(current_obs, self.weights)
        return 0.0

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
