import sys
import os
import time

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cg.game import battle_start, battle_select, battle_finish
from src.agent_factory import get_agent

# 対戦対象のエージェント
agent_types = ["random", "evolutionary", "mcts", "rl", "pytorch_rl", "enhanced_mcts"]

def run_game(p0_type, p1_type):
    """1試合の対戦を実行し、勝者(0:P0, 1:P1)とターン数を返します。"""
    p0_agent = get_agent(p0_type)
    p1_agent = get_agent(p1_type)
    
    # 決定論的挙動にするため、探索確率がある場合は0に設定
    for agent in [p0_agent, p1_agent]:
        if hasattr(agent, 'epsilon'):
            agent.epsilon = 0.0
            
    deck0 = p0_agent.read_deck_csv()
    deck1 = p1_agent.read_deck_csv()
    
    obs_dict, _ = battle_start(deck0, deck1)
    turn = 0
    winner = None
    try:
        while turn < 500:
            current_state = obs_dict.get("current")
            if current_state is not None:
                result = current_state.get("result", -1)
                if result != -1:
                    winner = result
                    break
            
            your_idx = obs_dict.get("current", {}).get("yourIndex", 0)
            if your_idx == 0:
                action = p0_agent.select_action(obs_dict)
            else:
                action = p1_agent.select_action(obs_dict)
            obs_dict = battle_select(action)
            turn += 1
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
    finally:
        try:
            battle_finish()
        except:
            pass
            
    return winner, turn

def main():
    print("==================================================")
    print(" AI総当たり評価バトル (各ペア5試合)")
    print("==================================================")
    
    # 総当たりのペアを作成
    pairs = []
    for i in range(len(agent_types)):
        for j in range(i+1, len(agent_types)):
            pairs.append((agent_types[i], agent_types[j]))
            
    total_games = sum(20 if (a1 == "rl" or a2 == "rl") else 5 for a1, a2 in pairs)
    game_count = 0
    
    # 成績データ構造の初期化
    stats = {t: {"wins": 0, "losses": 0, "draws": 0} for t in agent_types}
    matrix = {t1: {t2: {"wins": 0, "losses": 0, "draws": 0} for t2 in agent_types} for t1 in agent_types}
    
    start_time = time.time()
    
    for a1, a2 in pairs:
        print(f"\n--- {a1} vs {a2} (対戦開始) ---")
        # rlが絡む場合は20試合、それ以外は5試合行う
        if a1 == "rl" or a2 == "rl":
            arrangements = [(a1, a2)] * 10 + [(a2, a1)] * 10
        else:
            arrangements = [(a1, a2)] * 3 + [(a2, a1)] * 2
        
        for p0, p1 in arrangements:
            game_count += 1
            print(f"[{game_count}/{total_games}] {p0} (P0/先攻) vs {p1} (P1/後攻) ... ", end="", flush=True)
            
            g_winner, g_turns = run_game(p0, p1)
            
            if g_winner == 0:
                print(f"{p0}の勝利 ({g_turns}ターン)")
                stats[p0]["wins"] += 1
                stats[p1]["losses"] += 1
                matrix[p0][p1]["wins"] += 1
                matrix[p1][p0]["losses"] += 1
            elif g_winner == 1:
                print(f"{p1}の勝利 ({g_turns}ターン)")
                stats[p1]["wins"] += 1
                stats[p0]["losses"] += 1
                matrix[p1][p0]["wins"] += 1
                matrix[p0][p1]["losses"] += 1
            else:
                print(f"引き分け ({g_turns}ターン制限超過またはエラー)")
                stats[p0]["draws"] += 1
                stats[p1]["draws"] += 1
                matrix[p0][p1]["draws"] += 1
                matrix[p1][p0]["draws"] += 1
                
    elapsed = time.time() - start_time
    print("\n==================================================")
    print(f" 全対戦終了 (総所要時間: {elapsed:.1f}秒)")
    print("==================================================")
    
    # ランキングの集計
    ranking = []
    for t, s in stats.items():
        total = s["wins"] + s["losses"] + s["draws"]
        win_rate = s["wins"] / total if total > 0 else 0.0
        ranking.append((t, s["wins"], s["losses"], s["draws"], win_rate))
        
    ranking.sort(key=lambda x: x[4], reverse=True)
    
    print("\n【最終ランキング】")
    print(f"{'順位':<4}{'エージェント':<15}{'勝ち':<6}{'負け':<6}{'分':<6}{'勝率':<8}")
    for idx, (t, w, l, d, wr) in enumerate(ranking):
        print(f"{idx+1:<4}{t:<15}{w:<6}{l:<6}{d:<6}{wr*100:.1f}%")
        
    print("\n【対戦マトリクス (行 vs 列 / 勝-敗-分)】")
    # ヘッダー
    header = f"{'':<15}"
    for t in agent_types:
        header += f"{t[:6]:<10}"
    print(header)
    
    for t1 in agent_types:
        row = f"{t1:<15}"
        for t2 in agent_types:
            if t1 == t2:
                row += f"{'-':<10}"
            else:
                m = matrix[t1][t2]
                cell_text = f"{m['wins']}-{m['losses']}-{m['draws']}"
                row += f"{cell_text:<10}"
        print(row)
    print("==================================================")

if __name__ == "__main__":
    main()
