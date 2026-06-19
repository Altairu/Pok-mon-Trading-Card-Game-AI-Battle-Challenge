import random
import os
import torch
import torch.nn as nn
from cg.api import Observation, search_begin, search_step, search_end
from src.base_agent import BaseAgent
from src.evolutionary.ga import get_features

class ValueNetwork(nn.Module):
    """盤面状態の特徴量からその状態の価値を予測するニューラルネットワークモデル。"""
    def __init__(self, input_dim=15, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class PytorchRlAgent(BaseAgent):
    """PyTorchで定義された価値ネットワークを使用して行動選択を行う強化学習エージェント。"""
    def __init__(self, deck_path="deck.csv", model_path=None, epsilon=0.1):
        super().__init__(deck_path)
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = ValueNetwork().to(self.device)
        self.model.eval()
        
        if model_path is None:
            best_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolutionary", "pytorch_model_best.pth")
            normal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolutionary", "pytorch_model.pth")
            if os.path.exists(best_path):
                self.model_path = best_path
            else:
                self.model_path = normal_path
        else:
            self.model_path = model_path
            
        self.load_model()

    def load_model(self):
        """保存されたモデルパラメータをロードします。"""
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f"PyTorchモデルをロードしました: {os.path.basename(self.model_path)}")
            except Exception as e:
                print("モデルのロードに失敗しました。")

    def save_model(self):
        """現在のモデルパラメータをファイルに保存します。"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            print(f"PyTorchモデルを保存しました: {os.path.basename(self.model_path)}")
        except Exception as e:
            print("モデルの保存に失敗しました。")

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
            
            chosen_action = None
            if random.random() < self.epsilon:
                # 探索
                chosen_action = random.choice(candidates)
            else:
                # 活用
                valid_candidates = []
                feature_list = []
                for cand in candidates:
                    try:
                        next_state = search_step(root_state.searchId, cand)
                        features = get_features(next_state.observation)
                        feature_list.append(features)
                        valid_candidates.append(cand)
                    except:
                        continue
                
                if len(feature_list) > 0:
                    with torch.no_grad():
                        features_tensor = torch.tensor(feature_list, dtype=torch.float32).to(self.device)
                        values = self.model(features_tensor).view(-1).tolist()
                        
                        best_idx = 0
                        best_score = -float('inf')
                        for idx, val in enumerate(values):
                            if val > best_score:
                                best_score = val
                                best_idx = idx
                        chosen_action = valid_candidates[best_idx]
                else:
                    chosen_action = random.choice(candidates)
            
            if chosen_action is None:
                chosen_action = candidates[0]
                
            search_end()
            return chosen_action
            
        except Exception as e:
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
