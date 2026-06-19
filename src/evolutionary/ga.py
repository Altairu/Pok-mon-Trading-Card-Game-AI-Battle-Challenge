import random
import time
import os
from cg.api import Observation
from cg.game import battle_start, battle_select, battle_finish


# 評価関数の重みのデフォルト値
DEFAULT_WEIGHTS = [
    100.0,
    -100.0,
    5.0,
    -5.0,
    0.5,
    -0.5,
    10.0,
    -5.0,
    2.0,
    0.1,
    30.0,  # イワパレスのバトル場配置
    25.0,  # ヒーローマント貼付
    20.0,  # イワパレスのエネルギー3個以上（ジャンボアイス発動条件）
    15.0,  # ベンチのメガガルーラex
    10.0   # 相手バトルポケモンの逃げるコスト（ボスの指令による縛り）
]

# グローバルな学習ステータス（Flask から参照、操作されます）
training_status = {
    "running": False,
    "paused": False,
    "algorithm": "ga",
    "current_generation": 0,
    "total_generations": 50,
    "population_size": 20,
    "num_games_per_eval": 3,
    "best_score": 0.0,
    "scores_history": [],
    "best_weights": list(DEFAULT_WEIGHTS),
    "elapsed_time": 0.0,
    "timer_limit": 10800,  # デフォルト3時間 (秒)
    "stop_requested": False,
    "pause_requested": False,
    "message": "未開始"
}

def get_features(obs: Observation) -> list[float]:
    """Observation から盤面特徴量ベクトルを抽出します。"""
    if obs is None or obs.current is None:
        return [0.0] * 15
        
    state = obs.current
    your_idx = state.yourIndex
    opp_idx = 1 - your_idx
    
    me = state.players[your_idx]
    opp = state.players[opp_idx]
    
    # 1. 自分の取ったサイド枚数
    my_prizes_taken = 6 - len(me.prize)
    # 2. 相手の取ったサイド枚数
    opp_prizes_taken = 6 - len(opp.prize)
    
    def get_pokemon_stats(active_list: list, bench_list: list):
        count = 0
        total_hp = 0
        total_energy = 0
        
        # バトル場
        for p in active_list:
            if p is not None:
                count += 1
                total_hp += p.hp
                total_energy += len(p.energies)
                
        # ベンチ
        for p in bench_list:
            if p is not None:
                count += 1
                total_hp += p.hp
                total_energy += len(p.energies)
                
        return count, total_hp, total_energy

    my_count, my_hp, my_energy = get_pokemon_stats(me.active, me.bench)
    opp_count, opp_hp, opp_energy = get_pokemon_stats(opp.active, opp.bench)
    
    # 9. 自分の手札枚数
    my_hand_count = me.handCount if me.hand is None else len(me.hand)
    # 10. 自分のトラッシュ枚数
    my_discard_count = len(me.discard)

    # 11. iwapalace_in_active: 自分のバトル場がイワパレス (ID 345) か
    iwapalace_in_active = 0.0
    has_hero_cape_on_iwapalace = 0.0
    iwapalace_energy_ge_3 = 0.0
    
    for p in me.active:
        if p is not None and p.id == 345:
            iwapalace_in_active = 1.0
            # 12. has_hero_cape_on_iwapalace: イワパレスにヒーローマント (ID 1159) が付いているか
            if any(t.id == 1159 for t in p.tools):
                has_hero_cape_on_iwapalace = 1.0
            # 13. iwapalace_energy_ge_3: イワパレスにエネルギーが3個以上付いているか
            if len(p.energies) >= 3:
                iwapalace_energy_ge_3 = 1.0
                
    # 14. megagangaskhan_on_bench: ベンチにメガガルーラex (ID 756) がいるか
    megagangaskhan_on_bench = 0.0
    for p in me.bench:
        if p is not None and p.id == 756:
            megagangaskhan_on_bench = 1.0
            
    # 15. opp_active_retreat_cost: 相手のバトルポケモンの逃げるエネルギーコスト
    opp_retreat_cost = 0.0
    RETREAT_COSTS = {
        63: 2.0,   # タケルライコex
        96: 1.0,   # オーガポンex (みどりのめん)
        117: 1.0,  # オーガポンex (いしずえのめん)
        122: 1.0,  # シャリタツ
        344: 2.0,  # イシズマイ
        345: 3.0,  # イワパレス
        756: 3.0,  # メガガルーラex
    }
    for p in opp.active:
        if p is not None:
            opp_retreat_cost = RETREAT_COSTS.get(p.id, 1.0)
    
    return [
        float(my_prizes_taken),
        float(opp_prizes_taken),
        float(my_count),
        float(opp_count),
        float(my_hp),
        float(opp_hp),
        float(my_energy),
        float(opp_energy),
        float(my_hand_count),
        float(my_discard_count),
        float(iwapalace_in_active),
        float(has_hero_cape_on_iwapalace),
        float(iwapalace_energy_ge_3),
        float(megagangaskhan_on_bench),
        float(opp_retreat_cost)
    ]

def evaluate_state(obs: Observation, weights: list[float]) -> float:
    """指定された重みパラメータを用いて、現在の盤面状態の評価スコアを算出します。"""
    features = get_features(obs)
    return sum(f * w for f, w in zip(features, weights))

def evaluate_weights_by_match(weights: list[float], num_games: int = 3) -> float:
    """
    指定された重みを適用したエージェントをランダムエージェントと実際に対戦させ、
    勝率と獲得サイド枚数に基づいて適合度スコアを計算します。
    """
    from src.agent_factory import get_agent
    # 相手はランダムエージェント
    opp_agent = get_agent("random")
    # 自分のエージェント (重みを適用)
    me_agent = get_agent("evolutionary", weights=weights)
    
    deck0 = me_agent.read_deck_csv()
    deck1 = opp_agent.read_deck_csv()
    
    total_score = 0.0
    
    for game_idx in range(num_games):
        if training_status.get("stop_requested", False):
            break
            
        try:
            obs_dict, start_data = battle_start(deck0, deck1)
            
            turn = 0
            while turn < 500:  # 無限ループ防止
                # 一時停止の処理
                while training_status.get("pause_requested", False):
                    if training_status.get("stop_requested", False):
                        break
                    time.sleep(0.5)
                    
                if training_status.get("stop_requested", False):
                    break
                    
                current_state = obs_dict.get("current")
                if current_state is not None:
                    result = current_state.get("result", -1)
                    if result != -1:
                        # バトル終了
                        if result == 0:  # 自分が勝利
                            total_score += 10.0
                            
                        # サイドカード獲得数を評価に加味
                        me_prize = len(current_state["players"][0]["prize"])
                        opp_prize = len(current_state["players"][1]["prize"])
                        total_score += (6 - me_prize) * 2.0
                        total_score -= (6 - opp_prize) * 1.0
                        break
                        
                your_idx = obs_dict.get("current", {}).get("yourIndex", 0)
                if your_idx == 0:
                    action = me_agent.select_action(obs_dict)
                else:
                    action = opp_agent.select_action(obs_dict)
                    
                obs_dict = battle_select(action)
                turn += 1
                
            battle_finish()
            
        except Exception as e:
            # エラー発生時はその対戦はノーゲームとしてスキップ
            print(f"対戦シミュレーションエラー: {e}")
            try:
                battle_finish()
            except:
                pass
                
    if num_games > 0:
        avg_score = total_score / num_games
        # 0以上の正の適合度に補正
        return max(0.0, avg_score + 10.0)
    return 0.0

def crossover(parent1: list[float], parent2: list[float]) -> list[float]:
    """二点交叉を行います。"""
    child = []
    pt1 = random.randint(0, 8)
    pt2 = random.randint(pt1 + 1, 9)
    for i in range(10):
        if i < pt1 or i > pt2:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

def mutate(weights: list[float], scale: float = 5.0) -> list[float]:
    """突然変異を行います。"""
    mutated = []
    for w in weights:
        if random.random() < 0.1:  # 突然変異確率 10%
            mutated.append(w + random.gauss(0, scale))
        else:
            mutated.append(w)
    return mutated

def save_best_weights(weights: list[float]):
    """最良の重みをテキストファイルに書き出します。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "best_weights.txt")
    with open(file_path, "w") as f:
        f.write(",".join(map(str, weights)))

def load_best_weights() -> list[float]:
    """最良の重みをテキストファイルから読み込みます。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "best_weights.txt")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                content = f.read().strip()
                if content:
                    return list(map(float, content.split(",")))
        except Exception:
            pass
    return list(DEFAULT_WEIGHTS)

def train_ga_loop():
    """
    GA学習のメインループ。別スレッドで実行されることを想定しています。
    """
    global training_status
    
    training_status["running"] = True
    training_status["stop_requested"] = False
    training_status["pause_requested"] = False
    training_status["scores_history"] = []
    training_status["message"] = "学習開始"
    
    pop_size = training_status["population_size"]
    generations = training_status["total_generations"]
    num_games = training_status["num_games_per_eval"]
    timer_limit = training_status["timer_limit"]
    
    start_time = time.time()
    
    # 初期世代の生成（ファイルから読み込んだ最良重みをベースにする）
    base_weights = load_best_weights()
    population = [list(base_weights)]
    for _ in range(pop_size - 1):
        population.append([w + random.gauss(0, 15.0) for w in base_weights])
        
    for gen in range(generations):
        if training_status["stop_requested"]:
            training_status["message"] = "停止要求により中断"
            break
            
        elapsed = time.time() - start_time
        training_status["elapsed_time"] = elapsed
        
        # タイマーによる自動終了チェック
        if elapsed >= timer_limit:
            training_status["message"] = "設定時間に達したため学習を終了しました"
            break
            
        training_status["current_generation"] = gen + 1
        training_status["message"] = f"第 {gen+1} 世代の評価中..."
        
        # 各個体の適合度評価
        scores = []
        for idx, ind in enumerate(population):
            if training_status["stop_requested"]:
                break
            # UI側に何番目の個体を評価中か伝える
            training_status["message"] = f"第 {gen+1} 世代: 個体 {idx+1}/{pop_size} を対戦中..."
            score = evaluate_weights_by_match(ind, num_games=num_games)
            scores.append(score)
            
        if training_status["stop_requested"]:
            break
            
        # 最良個体の特定
        best_idx = 0
        best_score = -1.0
        for idx, score in enumerate(scores):
            if score > best_score:
                best_score = score
                best_idx = idx
                
        best_weights = population[best_idx]
        
        # 進捗をステータスに記録
        training_status["best_score"] = best_score
        training_status["scores_history"].append(best_score)
        training_status["best_weights"] = list(best_weights)
        
        # 世代ごとに最良の重みを自動保存
        save_best_weights(best_weights)
        
        # 次世代の作成（エリート保存 1個体 ＋ トーナメント選択による交叉・突然変異）
        new_population = [best_weights]
        
        # 突然変異スケールは世代が進むにつれて減少（アニーリング）
        mut_scale = max(1.0, 5.0 * (1.0 - (gen / generations)))
        
        while len(new_population) < pop_size:
            # トーナメントサイズ 3
            candidates = random.sample(list(zip(population, scores)), 3)
            parent1 = max(candidates, key=lambda x: x[1])[0]
            
            candidates = random.sample(list(zip(population, scores)), 3)
            parent2 = max(candidates, key=lambda x: x[1])[0]
            
            child = crossover(parent1, parent2)
            child = mutate(child, scale=mut_scale)
            new_population.append(child)
            
        population = new_population
        
    training_status["running"] = False
    if not training_status["stop_requested"]:
        training_status["message"] = "すべての学習ステップが完了しました"

def run_genetic_algorithm(obs: Observation) -> list[int] | None:
    """エージェントへのインターフェース（学習済みの最良重みをロードします）"""
    return load_best_weights()

class ReplayBuffer:
    """学習のための遷移データを保存する経験再生バッファ。"""
    def __init__(self, capacity):
        from collections import deque
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, reward, next_state, done):
        self.buffer.append((state, reward, next_state, done))
        
    def sample(self, batch_size):
        import torch
        state, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.tensor(state, dtype=torch.float32),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(next_state, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32))
                
    def __len__(self):
        return len(self.buffer)

def optimize_model(policy_net, target_net, optimizer, buffer, device, batch_size=64, gamma=0.95):
    """バッファからミニバッチを取り出してモデルパラメータを更新します。"""
    if len(buffer) < batch_size:
        return None
    import torch
    import torch.nn as nn
    states, rewards, next_states, dones = buffer.sample(batch_size)
    states = states.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)
    state_values = policy_net(states).squeeze(-1)
    with torch.no_grad():
        next_state_values = target_net(next_states).squeeze(-1)
        expected_state_values = rewards + (1.0 - dones) * gamma * next_state_values
    loss = nn.MSELoss()(state_values, expected_state_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_model_for_dashboard(policy_net, num_games=10):
    """ダッシュボード用の簡易モデル評価対戦を行います。"""
    from src.agent_factory import get_agent
    eval_agent = get_agent("pytorch_rl")
    eval_agent.model = policy_net
    eval_agent.epsilon = 0.0
    opp_agent = get_agent("random")
    deck0 = eval_agent.read_deck_csv()
    deck1 = opp_agent.read_deck_csv()
    wins = 0
    for _ in range(num_games):
        if training_status.get("stop_requested", False):
            break
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
        except:
            try:
                battle_finish()
            except:
                pass
    return wins / max(1, num_games)

def train_pytorch_rl_loop():
    """Webダッシュボードから起動される強化学習トレーニングループ。"""
    global training_status
    import torch
    import torch.optim as optim
    from src.agent_factory import get_agent
    from src.pytorch_rl_agent import ValueNetwork
    from cg.api import to_observation_class
    
    training_status["running"] = True
    training_status["stop_requested"] = False
    training_status["pause_requested"] = False
    training_status["scores_history"] = []
    training_status["best_score"] = 0.0
    training_status["message"] = "学習開始 (PyTorch RL)"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = ValueNetwork(input_dim=15).to(device)
    target_net = ValueNetwork(input_dim=15).to(device)
    
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evolutionary")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "pytorch_model.pth")
    best_path = os.path.join(model_dir, "pytorch_model_best.pth")
    
    if os.path.exists(model_path):
        try:
            policy_net.load_state_dict(torch.load(model_path, map_location=device))
        except:
            pass
            
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    buffer = ReplayBuffer(20000)
    
    total_games = training_status["total_generations"] * 100
    eval_interval = 100
    best_win_rate = -1.0
    total_losses = []
    
    start_time = time.time()
    
    for game_idx in range(total_games):
        if training_status["stop_requested"]:
            training_status["message"] = "停止要求により中断"
            break
            
        while training_status["pause_requested"]:
            if training_status["stop_requested"]:
                break
            time.sleep(0.5)
            
        elapsed = time.time() - start_time
        training_status["elapsed_time"] = elapsed
        if elapsed >= training_status["timer_limit"]:
            training_status["message"] = "制限時間に達したため終了しました"
            break
            
        training_status["current_generation"] = game_idx + 1
        training_status["total_generations"] = total_games
        
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
        
        try:
            obs_dict, _ = battle_start(deck0, deck1)
            turn = 0
            game_loss = []
            
            while turn < 500:
                if training_status["stop_requested"]:
                    break
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
                                
                                s_prime = [0.0] * 15
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
                
                # ターゲット更新間隔パラメータ用のダミー
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
                
            if game_idx % 10 == 0: # ターゲットネット更新
                target_net.load_state_dict(policy_net.state_dict())
                
            recent_loss = sum(total_losses[-50:]) / max(1, len(total_losses[-50:]))
            training_status["message"] = f"ゲーム {game_idx + 1}/{total_games} 進行中 | 相手: {opp_type} | ε: {p0_agent.epsilon:.2f} | 損失: {recent_loss:.4f}"
            
            if (game_idx + 1) % eval_interval == 0:
                training_status["message"] = f"ゲーム {game_idx + 1}/{total_games}: 評価テスト中..."
                win_rate = evaluate_model_for_dashboard(policy_net)
                training_status["best_score"] = win_rate * 100
                training_status["scores_history"].append(win_rate * 100)
                
                torch.save(policy_net.state_dict(), model_path)
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    torch.save(policy_net.state_dict(), best_path)
                    
        except Exception as e:
            try:
                battle_finish()
            except:
                pass
            training_status["message"] = f"ゲーム {game_idx + 1} でエラー発生: {str(e)}"
            
    training_status["running"] = False
    if not training_status["stop_requested"]:
        training_status["message"] = "強化学習モデルのトレーニングが完了しました"
