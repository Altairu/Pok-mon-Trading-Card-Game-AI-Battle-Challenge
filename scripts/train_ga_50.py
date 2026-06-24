import sys
import os

# Windows環境での文字コードエラー（Pokémonの"é"など）を回避するために標準出力をUTF-8に再設定します
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import random
import time
import argparse
import math

# プロジェクトルートのパスを追加してインポート可能にします
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cg.game import battle_start, battle_select, battle_finish
from src.agent_factory import get_agent
from src.enhanced_mcts_agent import DEFAULT_WEIGHTS_54, load_weights_54

# 最良重みの保存先パスを定義（enhanced_mcts_agent.pyと同一）
WEIGHTS_FILE_54 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "evolutionary", "best_weights_54.txt"
)

def clip_weights_54(weights: list[float]) -> list[float]:
    """54次元パラメータのうち、探索パラメータ（50〜53次元）を適切な範囲に丸めます。"""
    clipped = list(weights)
    # bias_weight (50)
    clipped[50] = max(0.0001, min(0.5, clipped[50]))
    # UCB c_param early (51)
    clipped[51] = max(0.05, min(5.0, clipped[51]))
    # UCB c_param mid (52)
    clipped[52] = max(0.05, min(4.0, clipped[52]))
    # UCB c_param late (53)
    clipped[53] = max(0.01, min(3.0, clipped[53]))
    return clipped

def evaluate_weights_by_match(weights: list[float], opponent_type: str = "dynamic", num_games: int = 3) -> tuple[float, float]:
    """
    指定された54次元の重みを適用した enhanced_mcts エージェントを対戦させ、
    勝率と獲得サイド枚数に基づいて適合度スコアを計算します。
    自己対戦（Co-evolution）および多様なエージェントとの対戦（多面対戦）に対応。
    """
    # 自分のエージェント (重みを適用、評価中は思考時間を0.2秒に短縮して超高速化)
    me_agent = get_agent("enhanced_mcts", weights=weights, time_limit_override=0.2)
    deck0 = me_agent.read_deck_csv()
    
    total_score = 0.0
    wins = 0
    
    for game_idx in range(num_games):
        try:
            # 対戦相手の決定
            if opponent_type == "dynamic":
                r_val = random.random()
                if r_val < 0.30:
                    # 自己対戦：歴代最良パラメータをロードして対戦
                    best_weights = load_weights_54()
                    opp_agent = get_agent("enhanced_mcts", weights=best_weights, time_limit_override=0.2)
                    opp_label = "self-play (best)"
                else:
                    # 多面対戦：主要エージェントからランダム選択
                    opp_type = random.choice(["mcts", "rl", "evolutionary", "random"])
                    opp_agent = get_agent(opp_type)
                    opp_label = opp_type
            else:
                opp_agent = get_agent(opponent_type)
                opp_label = opponent_type

            deck1 = opp_agent.read_deck_csv()

            # 先攻・後攻を交互に入れ替える
            if game_idx % 2 == 0:
                p0_agent, p1_agent = me_agent, opp_agent
                my_p_idx = 0
            else:
                p0_agent, p1_agent = opp_agent, me_agent
                my_p_idx = 1
                
            obs_dict, _ = battle_start(deck0, deck1)
            
            turn = 0
            while turn < 300:  # 無限ループ防止
                current_state = obs_dict.get("current")
                if current_state is not None:
                    result = current_state.get("result", -1)
                    if result != -1:
                        # ゲーム終了
                        is_win = (result == my_p_idx)
                        if is_win:
                            wins += 1
                            total_score += 1000.0  # 勝利ボーナス
                        
                        # 終了時のサイド差を細かく反映
                        me_prize = len(current_state["players"][my_p_idx]["prize"])
                        opp_prize = len(current_state["players"][1 - my_p_idx]["prize"])
                        total_score += (6 - me_prize) * 100.0
                        total_score -= (6 - opp_prize) * 50.0
                        
                        # 早期決着ボーナス（勝利時のみ）
                        if is_win:
                            total_score += max(0, 100 - turn) * 2.0
                        break
                        
                your_idx = obs_dict.get("current", {}).get("yourIndex", 0)
                if your_idx == 0:
                    action = p0_agent.select_action(obs_dict)
                else:
                    action = p1_agent.select_action(obs_dict)
                    
                obs_dict = battle_select(action)
                turn += 1
                
            battle_finish()
            
        except Exception as e:
            # エラー発生時は安全にクリーンアップしてスキップ
            print(f"    [Warning] 対戦シミュレーションエラー (相手: {opp_label}): {e}")
            try:
                battle_finish()
            except:
                pass
                
    if num_games > 0:
        # スコアを正規化
        avg_score = total_score / num_games
        win_rate = (wins / num_games) * 100.0
        return max(0.0, avg_score), win_rate
    return 0.0, 0.0

def crossover_blx_alpha(parent1: list[float], parent2: list[float], alpha: float = 0.5) -> list[float]:
    """実数値GAで効果的なブレンド交叉(BLX-alpha)を行います。"""
    child = []
    for p1, p2 in zip(parent1, parent2):
        x_min = min(p1, p2)
        x_max = max(p1, p2)
        d = x_max - x_min
        # 範囲 [x_min - alpha * d, x_max + alpha * d] からサンプリング
        val = random.uniform(x_min - alpha * d, x_max + alpha * d)
        child.append(val)
    return child

def mutate_gauss(weights: list[float], mut_prob: float = 0.15, scale: float = 10.0) -> list[float]:
    """ガウス突然変異を行います。探索パラメータの絶対値の大きさに合わせた個別のノイズスケールを適用します。"""
    mutated = []
    for idx, w in enumerate(weights):
        if random.random() < mut_prob:
            # 探索パラメータは評価パラメータに比べて値が非常に小さいため、小さなスケールで変異させる
            if idx >= 50:
                param_scale = max(0.01, abs(w) * 0.2)
                mutated.append(w + random.gauss(0, param_scale))
            else:
                mutated.append(w + random.gauss(0, scale))
        else:
            mutated.append(w)
    return mutated

def save_best_weights_54(weights: list[float]):
    """最良の54次元重みパラメータをファイルに上書き保存します。"""
    os.makedirs(os.path.dirname(WEIGHTS_FILE_54), exist_ok=True)
    with open(WEIGHTS_FILE_54, "w") as f:
        f.write(",".join(map(str, weights)))
    print(f"\n=> 最良のパラメータを自動保存しました: {WEIGHTS_FILE_54}")

def main():
    parser = argparse.ArgumentParser(description="54次元評価/探索パラメータGA自動学習（AI教育）")
    parser.add_argument("--generations", type=int, default=10, help="総世代数（デフォルト10）")
    parser.add_argument("--pop-size", type=int, default=12, help="個体群サイズ（デフォルト12、偶数推奨）")
    parser.add_argument("--games", type=int, default=3, help="1個体あたりの評価対戦数（デフォルト3）")
    parser.add_argument("--opponents", type=str, default="dynamic", help="対戦相手（mcts, random, evolutionaryなど、dynamicは自己対戦/多面対戦）")
    parser.add_argument("--mutation-prob", type=float, default=0.15, help="突然変異確率（デフォルト0.15）")
    args = parser.parse_args()

    print("=" * 60)
    print(" 54次元評価/探索パラメータ自動学習（AIの教育プロセス）")
    print(f" 設定: 世代数={args.generations}, 個体数={args.pop_size}, 評価対戦数={args.games}")
    print(f" 対戦相手={args.opponents}, 保存先={WEIGHTS_FILE_54}")
    print("=" * 60)

    start_time = time.time()
    
    # 1. 初期個体群の生成
    base_weights = load_weights_54()
    population = [list(base_weights)]  # エリート候補として基準個体を残す
    
    # 基準個体に対して適度なばらつきを加える
    for i in range(args.pop_size - 1):
        ind = []
        for idx, w in enumerate(base_weights):
            if idx >= 50:
                # 探索パラメータ用の小さなノイズ
                noise_scale = max(0.01, abs(w) * 0.20)
                ind.append(w + random.gauss(0, noise_scale))
            else:
                # 盤面評価パラメータ用の大きめのノイズ
                noise_scale = max(2.0, abs(w) * 0.15)
                ind.append(w + random.gauss(0, noise_scale))
        # 境界範囲に丸め
        ind = clip_weights_54(ind)
        population.append(ind)

    best_overall_weights = list(base_weights)
    best_overall_fitness = -1.0
    best_overall_win_rate = 0.0

    # 2. GAメインループ
    for gen in range(args.generations):
        gen_start_time = time.time()
        print(f"\n--- 第 {gen+1} / {args.generations} 世代の評価中 ---")
        
        fitness_scores = []
        win_rates = []
        
        for idx, ind in enumerate(population):
            fit, wr = evaluate_weights_by_match(ind, opponent_type=args.opponents, num_games=args.games)
            fitness_scores.append(fit)
            win_rates.append(wr)
            # 現在の探索パラメータ（bias_weight, c_early, c_mid, c_late）をログ出力して進行を確認しやすくする
            print(f"  個体 {idx+1}/{args.pop_size} -> 適合度: {fit:.1f}, 勝率: {wr:.1f}% | (bias={ind[50]:.4f}, c_early={ind[51]:.2f}, c_mid={ind[52]:.2f}, c_late={ind[53]:.2f})")

        # 世代内での最良個体の特定
        best_gen_idx = max(range(args.pop_size), key=lambda i: fitness_scores[i])
        best_gen_fitness = fitness_scores[best_gen_idx]
        best_gen_win_rate = win_rates[best_gen_idx]
        best_gen_weights = population[best_gen_idx]

        # 歴代最良個体の更新判定
        if best_gen_fitness > best_overall_fitness:
            best_overall_fitness = best_gen_fitness
            best_overall_win_rate = best_gen_win_rate
            best_overall_weights = list(best_gen_weights)
            # 保存
            save_best_weights_54(best_overall_weights)

        gen_elapsed = time.time() - gen_start_time
        print(f"第 {gen+1} 世代完了! 最良適合度: {best_gen_fitness:.1f} (勝率: {best_gen_win_rate:.1f}%) | 世代所要時間: {gen_elapsed:.1f}秒")

        # 3. 次世代の個体群作成（世代交代）
        new_population = [list(best_overall_weights)]  # エリート保存

        # 焼きなまし突然変異スケールの調整
        annealing_factor = max(0.2, 1.0 - (gen / args.generations))
        mut_scale = 12.0 * annealing_factor

        # トーナメントサイズ
        t_size = min(3, len(population))

        while len(new_population) < args.pop_size:
            # トーナメント選択
            cand1 = random.sample(list(zip(population, fitness_scores)), t_size)
            parent1 = max(cand1, key=lambda x: x[1])[0]
            
            cand2 = random.sample(list(zip(population, fitness_scores)), t_size)
            parent2 = max(cand2, key=lambda x: x[1])[0]

            # 交叉
            child = crossover_blx_alpha(parent1, parent2, alpha=0.4)
            # 突然変異
            child = mutate_gauss(child, mut_prob=args.mutation_prob, scale=mut_scale)
            # 境界範囲に丸め
            child = clip_weights_54(child)
            new_population.append(child)

        population = new_population

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(" 学習（教育プロセスの完了）完了")
    print(f" 歴代最良適合度: {best_overall_fitness:.1f}")
    print(f" 歴代最良対戦勝率: {best_overall_win_rate:.1f}%")
    print(f" 最良パラメータ: (bias={best_overall_weights[50]:.4f}, c_early={best_overall_weights[51]:.2f}, c_mid={best_overall_weights[52]:.2f}, c_late={best_overall_weights[53]:.2f})")
    print(f" 総学習所要時間: {total_elapsed / 60:.1f} 分")
    print("=" * 60)

if __name__ == "__main__":
    main()
