import sys
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cg.game import battle_start, battle_select, battle_finish
from cg.api import Observation, to_observation_class
from src.agent_factory import get_agent
from src.pytorch_rl_agent import PytorchRlAgent, ValueNetwork
from src.evolutionary.ga import get_features

TOTAL_GAMES = 10000
BATCH_SIZE = 64
GAMMA = 0.95
LR = 0.001
BUFFER_SIZE = 20000
TARGET_UPDATE_INTERVAL = 10
EVAL_INTERVAL = 500

class ReplayBuffer:
    """学習のための遷移データを保存する経験再生バッファ。"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, reward, next_state, done):
        self.buffer.append((state, reward, next_state, done))
        
    def sample(self, batch_size):
        state, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.tensor(state, dtype=torch.float32),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(next_state, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32))
                
    def __len__(self):
        return len(self.buffer)

def optimize_model(policy_net, target_net, optimizer, buffer, device):
    """バッファからミニバッチを取り出してモデルのパラメータを更新します。"""
    if len(buffer) < BATCH_SIZE:
        return None
        
    states, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
    states = states.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)
    
    state_values = policy_net(states).squeeze(-1)
    
    with torch.no_grad():
        next_state_values = target_net(next_states).squeeze(-1)
        expected_state_values = rewards + (1.0 - dones) * GAMMA * next_state_values
        
    loss = nn.MSELoss()(state_values, expected_state_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate_model(policy_net, device, num_games=20):
    """学習中のモデルとランダムエージェントを対戦させて勝率を評価します。"""
    print(f"\n--- 評価テストを実行中 ({num_games} ゲーム) ---")
    eval_agent = get_agent("pytorch_rl")
    eval_agent.model = policy_net
    eval_agent.epsilon = 0.0
    
    opp_agent = get_agent("random")
    
    deck0 = eval_agent.read_deck_csv()
    deck1 = opp_agent.read_deck_csv()
    
    wins = 0
    for _ in range(num_games):
        try:
            obs_dict, _ = battle_start(deck0, deck1)
            turn = 0
            while turn < 500:
                current_state = obs_dict.get("current")
                if current_state is not None:
                    result = current_state.get("result", -1)
                    if result != -1:
                        if result == 0:
                            wins += 1
                        break
                your_idx = obs_dict.get("current", {}).get("yourIndex", 0)
                if your_idx == 0:
                    action = eval_agent.select_action(obs_dict)
                else:
                    action = opp_agent.select_action(obs_dict)
                obs_dict = battle_select(action)
                turn += 1
            battle_finish()
        except Exception as e:
            try:
                battle_finish()
            except:
                pass
            
    win_rate = wins / num_games
    print(f"評価テスト結果: 勝率 {win_rate * 100:.1f}%\n")
    return win_rate

def train():
    """10000回の対決を通して強化学習モデルのトレーニングを行うメインループ。"""
    # GPUの互換性エラーを回避するためCPUで実行します
    device = torch.device("cpu")
    print(f"デバイス: {device}")
    
    policy_net = ValueNetwork().to(device)
    target_net = ValueNetwork().to(device)
    
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "evolutionary")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "pytorch_model.pth")
    
    if os.path.exists(model_path):
        try:
            policy_net.load_state_dict(torch.load(model_path, map_location=device))
            print("既存のモデルパラメータをロードしました。")
        except Exception:
            print("既存モデルのロードに失敗しました。")
            
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    best_win_rate = -1.0
    total_losses = []
    
    try:
        for game_idx in range(TOTAL_GAMES):
            rand_val = random.random()
            if rand_val < 0.8:
                opp_type = "pytorch_rl"
            elif rand_val < 0.95:
                opp_type = "evolutionary"
            else:
                opp_type = "random"
                
            p0_agent = get_agent("pytorch_rl", epsilon=max(0.01, 0.2 - (game_idx / 5000)))
            p0_agent.model = policy_net
            
            p1_agent = get_agent(opp_type)
            if opp_type == "pytorch_rl":
                p1_agent.model = policy_net
                p1_agent.epsilon = p0_agent.epsilon
                
            is_learning_agent = {0: True, 1: (opp_type == "pytorch_rl")}
            
            deck0 = p0_agent.read_deck_csv()
            deck1 = p1_agent.read_deck_csv()
            
            prev_state = {0: None, 1: None}
            prev_prizes = {0: None, 1: None}
            prev_opp_prizes = {0: None, 1: None}
            
            obs_dict, _ = battle_start(deck0, deck1)
            
            turn = 0
            game_loss = []
            
            while turn < 500:
                current_state = obs_dict.get("current")
                if current_state is not None:
                    result = current_state.get("result", -1)
                    if result != -1:
                        for p_idx in [0, 1]:
                            if is_learning_agent[p_idx] and prev_state[p_idx] is not None:
                                is_win = (result == p_idx)
                                reward = 10.0 if is_win else -10.0
                                
                                me_prize = len(current_state["players"][p_idx]["prize"])
                                opp_prize = len(current_state["players"][1 - p_idx]["prize"])
                                prize_taken = prev_prizes[p_idx] - me_prize
                                opp_prize_taken = prev_opp_prizes[p_idx] - opp_prize
                                reward += prize_taken * 2.0 - opp_prize_taken * 2.0
                                
                                # 山札切れ自滅防止ペナルティ
                                my_deck_len = current_state["players"][p_idx]["deckCount"]
                                if my_deck_len <= 5:
                                    reward -= (6.0 - my_deck_len) * 1.5
                                
                                s_prime = [0.0] * 10
                                buffer.push(prev_state[p_idx], reward, s_prime, 1.0)
                        break
                        
                your_idx = obs_dict.get("current", {}).get("yourIndex", 0)
                
                if is_learning_agent[your_idx] and prev_state[your_idx] is not None:
                    me_prize = len(current_state["players"][your_idx]["prize"])
                    opp_prize = len(current_state["players"][1 - your_idx]["prize"])
                    prize_taken = prev_prizes[your_idx] - me_prize
                    opp_prize_taken = prev_opp_prizes[your_idx] - opp_prize
                    reward = prize_taken * 2.0 - opp_prize_taken * 2.0
                    
                    # 山札切れ自滅防止ペナルティ
                    my_deck_len = current_state["players"][your_idx]["deckCount"]
                    if my_deck_len <= 5:
                        reward -= (6.0 - my_deck_len) * 1.5
                    
                    obs_obj = to_observation_class(obs_dict)
                    s_prime = get_features(obs_obj)
                    buffer.push(prev_state[your_idx], reward, s_prime, 0.0)
                    
                obs_obj = to_observation_class(obs_dict)
                prev_state[your_idx] = get_features(obs_obj)
                prev_prizes[your_idx] = len(current_state["players"][your_idx]["prize"])
                prev_opp_prizes[your_idx] = len(current_state["players"][1 - your_idx]["prize"])
                
                if your_idx == 0:
                    action = p0_agent.select_action(obs_dict)
                else:
                    action = p1_agent.select_action(obs_dict)
                    
                obs_dict = battle_select(action)
                
                loss = optimize_model(policy_net, target_net, optimizer, buffer, device)
                if loss is not None:
                    game_loss.append(loss)
                    
                turn += 1
                
            battle_finish()
            
            if len(game_loss) > 0:
                avg_loss = sum(game_loss) / len(game_loss)
                total_losses.append(avg_loss)
            else:
                avg_loss = 0.0
                
            if game_idx % TARGET_UPDATE_INTERVAL == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            if (game_idx + 1) % 50 == 0:
                recent_loss = sum(total_losses[-50:]) / max(1, len(total_losses[-50:]))
                print(f"ゲーム {game_idx + 1}/{TOTAL_GAMES} 完了 | 相手: {opp_type} | ε: {p0_agent.epsilon:.3f} | 平均損失: {recent_loss:.5f}")
                
            if (game_idx + 1) % EVAL_INTERVAL == 0:
                win_rate = evaluate_model(policy_net, device)
                torch.save(policy_net.state_dict(), model_path)
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_path = os.path.join(model_dir, "pytorch_model_best.pth")
                    torch.save(policy_net.state_dict(), best_path)
                    print(f"最高勝率を更新しました: {win_rate * 100:.1f}% | ベストモデルを保存しました。")
                    
    except KeyboardInterrupt:
        print("\n学習が手動で中断されました。現在のモデルを保存します。")
        torch.save(policy_net.state_dict(), model_path)
        print("モデルを保存しました。")

if __name__ == "__main__":
    train()
