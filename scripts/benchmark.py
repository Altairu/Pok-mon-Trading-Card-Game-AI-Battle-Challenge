"""
enhanced_mcts vs 既存エージェントのベンチマークスクリプト。

各エージェントと複数試合を行い、勝率を比較する。
ローカル実行用に enhanced_mcts の思考時間を短縮できる。

使い方:
  python scripts/benchmark.py                # デフォルト（1秒/ターン、各3試合）
  python scripts/benchmark.py --time 2       # 2秒/ターン
  python scripts/benchmark.py --games 5      # 各5試合
  python scripts/benchmark.py --opponents mcts rl  # 対象絞り込み
"""

import sys
import os
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cg.game import battle_start, battle_select, battle_finish
from src.agent_factory import get_agent


def run_game(p0_type: str, p1_type: str, p0_kwargs: dict, p1_kwargs: dict,
             max_turns: int = 300) -> tuple[int | None, int, float]:
    """1試合実行して (勝者インデックス, ターン数, 経過秒) を返す。"""
    p0 = get_agent(p0_type, **p0_kwargs)
    p1 = get_agent(p1_type, **p1_kwargs)

    # RL系は探索をオフにする
    for ag in [p0, p1]:
        if hasattr(ag, "epsilon"):
            ag.epsilon = 0.0

    deck0 = p0.read_deck_csv()
    deck1 = p1.read_deck_csv()

    obs_dict, _ = battle_start(deck0, deck1)
    t_start = time.time()
    winner = None

    try:
        for turn in range(max_turns):
            current = obs_dict.get("current")
            if current is not None and current.get("result", -1) != -1:
                winner = current["result"]
                break

            your_idx = obs_dict.get("current", {}).get("yourIndex", 0)
            action = p0.select_action(obs_dict) if your_idx == 0 else p1.select_action(obs_dict)
            obs_dict = battle_select(action)
    except Exception as e:
        print(f"    エラー: {e}")
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    elapsed = time.time() - t_start
    return winner, turn, elapsed


def benchmark(time_limit: float, num_games: int, opponents: list[str]):
    print("=" * 60)
    print(f" enhanced_mcts ベンチマーク")
    print(f" 思考時間: {time_limit}秒/ターン  試合数/相手: {num_games}")
    print("=" * 60)

    results = {}

    for opp_type in opponents:
        print(f"\n--- enhanced_mcts vs {opp_type} ---")
        wins = losses = draws = 0
        total_time = 0.0

        for i in range(num_games):
            # 先攻・後攻を交互に入れ替える
            if i % 2 == 0:
                p0_type, p1_type = "enhanced_mcts", opp_type
                p0_kw = {"time_limit_override": time_limit}
                p1_kw = {}
            else:
                p0_type, p1_type = opp_type, "enhanced_mcts"
                p0_kw = {}
                p1_kw = {"time_limit_override": time_limit}

            winner, turns, elapsed = run_game(p0_type, p1_type, p0_kw, p1_kw)
            total_time += elapsed

            # enhanced_mcts が先攻なら winner==0 で勝ち、後攻なら winner==1 で勝ち
            if winner is None:
                draws += 1
                label = "引き分け"
                my_result = "draw"
            elif (i % 2 == 0 and winner == 0) or (i % 2 == 1 and winner == 1):
                wins += 1
                label = "enhanced_mcts 勝利"
                my_result = "win"
            else:
                losses += 1
                label = f"{opp_type} 勝利"
                my_result = "loss"

            role = "先攻" if i % 2 == 0 else "後攻"
            print(f"  [{i+1}/{num_games}] {role}: {label} ({turns}ターン, {elapsed:.0f}秒)")

        total = wins + losses + draws
        win_rate = wins / total * 100 if total > 0 else 0
        avg_time = total_time / num_games if num_games > 0 else 0

        print(f"  結果: {wins}勝 {losses}敗 {draws}分  勝率: {win_rate:.1f}%  平均: {avg_time:.0f}秒/試合")
        results[opp_type] = {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}

    # サマリー
    print("\n" + "=" * 60)
    print(" サマリー")
    print("=" * 60)
    print(f" {'相手':<15} {'勝':<5} {'敗':<5} {'分':<5} {'勝率'}")
    for opp, r in results.items():
        print(f" {opp:<15} {r['wins']:<5} {r['losses']:<5} {r['draws']:<5} {r['win_rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="enhanced_mcts ベンチマーク")
    parser.add_argument("--time", type=float, default=1.0,
                        help="enhanced_mcts の思考時間（秒/ターン、デフォルト1秒）")
    parser.add_argument("--games", type=int, default=3,
                        help="各相手との試合数（デフォルト3）")
    parser.add_argument("--opponents", nargs="+",
                        default=["random", "mcts", "rl", "evolutionary"],
                        help="対戦相手のエージェントタイプ")
    args = parser.parse_args()

    benchmark(args.time, args.games, args.opponents)


if __name__ == "__main__":
    main()
