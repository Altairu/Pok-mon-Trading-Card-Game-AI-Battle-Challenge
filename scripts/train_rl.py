import sys
import os
import random
import time

# プロジェクトルートのパスを追加してインポート可能にします
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cg.game import battle_start, battle_select, battle_finish
from src.agent_factory import get_agent
from src.rl_agent import RlAgent

TOTAL_GAMES = 1000
# 自己対戦比率 60%
SELF_PLAY_RATIO = 0.6

def run_game(p0, p1):
    """1試合の対戦を実行し、勝者(0:P0, 1:P1)とターン数を返します。"""
    deck0 = p0.read_deck_csv()
    deck1 = p1.read_deck_csv()
    
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
                action = p0.select_action(obs_dict)
            else:
                action = p1.select_action(obs_dict)
            obs_dict = battle_select(action)
            turn += 1
    except Exception as e:
        print(f"\n対戦中にエラーが発生しました: {e}")
    finally:
        try:
            battle_finish()
        except:
            pass
            
    return winner, turn

def main():
    print("==================================================")
    print(" RLエージェント専用トレーニングスクリプト (1000ゲーム)")
    print("==================================================")
    
    # 進行状況の管理
    wins = 0
    losses = 0
    draws = 0
    
    # 50ゲームごとの成績計測用
    interval_wins = 0
    interval_games = 0
    
    start_time = time.time()
    
    for game_idx in range(1, TOTAL_GAMES + 1):
        # epsilonの減衰 (初期 0.2 から 最終 0.02)
        epsilon = max(0.02, 0.2 - (game_idx / TOTAL_GAMES) * 0.18)
        
        # 対戦相手の決定
        r = random.random()
        if r < SELF_PLAY_RATIO:
            # 自己対戦 (rl vs rl)
            p0 = get_agent("rl", epsilon=epsilon)
            p1 = get_agent("rl", epsilon=epsilon)
            opponent_name = "rl (自己対戦)"
        else:
            # 他のエージェントとの対戦
            p0 = get_agent("rl", epsilon=epsilon)
            opp_choice = random.random()
            if opp_choice < 0.5:
                p1 = get_agent("evolutionary")
                opponent_name = "evolutionary"
            elif opp_choice < 0.8:
                p1 = get_agent("mcts")
                opponent_name = "mcts"
            else:
                p1 = get_agent("random")
                opponent_name = "random"
        
        # 先攻後攻をランダムに入れ替え
        p0_is_rl = isinstance(p0, RlAgent)
        p1_is_rl = isinstance(p1, RlAgent)
        
        if random.random() < 0.5:
            # 入れ替え
            p0, p1 = p1, p0
            p0_is_rl, p1_is_rl = p1_is_rl, p0_is_rl
            
        winner_idx, turns = run_game(p0, p1)
        
        # 勝敗結果の記録
        is_self_play = p0_is_rl and p1_is_rl
        
        if winner_idx == 0:
            if p0_is_rl:
                if not is_self_play:
                    wins += 1
                    interval_wins += 1
            else:
                losses += 1
        elif winner_idx == 1:
            if p1_is_rl:
                if not is_self_play:
                    wins += 1
                    interval_wins += 1
            else:
                losses += 1
        else:
            if not is_self_play:
                draws += 1
                
        if not is_self_play:
            interval_games += 1
            
        # ゲーム終了後にRLエージェントの重みを明示的に保存
        if p0_is_rl:
            p0.save_weights()
        if p1_is_rl:
            p1.save_weights()
            
        # 進行状況の表示
        if game_idx % 10 == 0:
            elapsed = time.time() - start_time
            games_per_sec = game_idx / elapsed if elapsed > 0 else 0.0
            print(f"ゲーム {game_idx}/{TOTAL_GAMES} 完了 / 相手: {opponent_name} / {turns}ターン / 速度: {games_per_sec:.2f} ゲーム/秒")
            
        # 50ゲームごとの重みパラメータと戦績の出力
        if game_idx % 50 == 0:
            # 最新の重みを読み込んで表示
            rl_ref = get_agent("rl")
            weights_str = ", ".join(f"{w:.2f}" for w in rl_ref.weights)
            print("\n--------------------------------------------------")
            print(f"【進捗報告 - {game_idx}ゲーム経過】")
            if interval_games > 0:
                win_rate = interval_wins / interval_games * 100
                print(f"直近50ゲーム内の他エージェント戦勝率: {win_rate:.1f}% ({interval_wins}勝/{interval_games}戦)")
            print(f"現在の重みパラメータ:")
            print(f"[{weights_str}]")
            print("--------------------------------------------------\n")
            
            # 区間成績のクリア
            interval_wins = 0
            interval_games = 0

    # 最終保存
    final_rl = get_agent("rl")
    final_rl.save_weights()
    print("トレーニングが正常に終了し、最終重みが保存されました。")

if __name__ == "__main__":
    main()
