"""
強化MCTSエージェントの簡易動作確認スクリプト。
enhanced_mcts vs random を数回対戦させ、正常に動くかチェックする。
"""
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cg.game import battle_start, battle_select, battle_finish
from src.agent_factory import get_agent


def run_game(p0_type: str, p1_type: str, game_idx: int):
    p0 = get_agent(p0_type, time_limit_override=0.1) if p0_type == "enhanced_mcts" else get_agent(p0_type)
    p1 = get_agent(p1_type, time_limit_override=0.1) if p1_type == "enhanced_mcts" else get_agent(p1_type)

    deck0 = p0.read_deck_csv()
    deck1 = p1.read_deck_csv()

    obs_dict, _ = battle_start(deck0, deck1)
    turn = 0
    winner = None
    t_start = time.time()

    try:
        while turn < 200:
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
        print(f"エラー: {e}")
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    elapsed = time.time() - t_start
    result_str = {0: f"{p0_type}勝利", 1: f"{p1_type}勝利"}.get(winner, "引き分け")
    print(f"[ゲーム{game_idx+1}] {p0_type}(P0) vs {p1_type}(P1) → {result_str} ({turn}ターン, {elapsed:.1f}秒)")
    return winner


def main():
    print("=== 強化MCTSエージェント 動作確認テスト ===\n")

    # enhanced_mcts vs random を3試合
    results = {"wins": 0, "losses": 0, "draws": 0}
    for i in range(3):
        w = run_game("enhanced_mcts", "random", i)
        if w == 0:
            results["wins"] += 1
        elif w == 1:
            results["losses"] += 1
        else:
            results["draws"] += 1

    print(f"\n結果: {results['wins']}勝 {results['losses']}敗 {results['draws']}分")
    total = sum(results.values())
    if total > 0:
        print(f"勝率: {results['wins']/total*100:.1f}%")


if __name__ == "__main__":
    main()
