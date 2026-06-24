"""
強化MCTSエージェント

10分の時間制限をフル活用する、コンテキスト対応型の改良MCTSエージェント。
主な改善点：
  - 特徴量を50次元に拡張
  - ゲームフェーズ（序盤/中盤/終盤）による戦略切り替え
  - ランダムプレイアウトの代わりにヒューリスティックガイドプレイアウトを採用
  - Progressive Bias付きUCBで評価関数スコアをMCTS選択に活用
  - ログからの相手カード追跡によるデッキ推定の改善
  - 時間制限ベースの反復深化（イテレーション上限をほぼ無制限に）
"""

import random
import math
import time
import os
from collections import defaultdict
from cg.api import (
    Observation, State, PlayerState, Log, LogType, OptionType, SelectType, AreaType,
    search_begin, search_step, search_release, search_end
)
from src.base_agent import BaseAgent


# ---------------------- デッキ定義 ----------------------

# イワパレス軸デッキ（自分のデッキ）のカードID一覧
MY_DECK_IDS = [
    344, 344, 344, 344,   # イシズマイ x4
    345, 345, 345,         # イワパレス x3
    756, 756,              # メガガルーラex x2
    117,                   # オーガポンex x1
    1086, 1086,            # カード x2
    1121, 1121,            # カード x2
    1122, 1122, 1122, 1122, # x4
    1123,                  # x1
    1147, 1147, 1147, 1147, # x4
    1120, 1120, 1120, 1120, # x4
    1159,                  # ヒーローマント x1
    1227, 1227, 1227, 1227, # x4
    1182, 1182, 1182, 1182, # x4
    1219, 1219, 1219, 1219, # x4
    1197,                  # x1
    1225, 1225, 1225,      # x3
    1212,                  # x1
    1257,                  # x1
    18, 18, 18, 18,        # エネルギー x4
    14, 14, 14, 14,        # エネルギー x4
    11, 11, 11, 11,        # エネルギー x4
    20, 20,                # エネルギー x2
]

# 汎用的に強力なたねポケモンのIDリスト（相手デッキ補填用）
# 実際の大会で多用されるたねポケモン
STRONG_BASIC_POKEMON_IDS = [63, 96, 117, 344, 1]

# 汎用カード（相手デッキ補填用）- たねポケモンを最低4枚含む
GENERIC_CARD_IDS = [
    # たねポケモン（必須：これがないとシミュレータがセットアップできない）
    63, 63,              # タケルライコex x2
    96, 117,             # オーガポンex x2
    # その他汎用カード
    1086, 1086, 1086, 1086,
    1121, 1121, 1121, 1121,
    1182, 1182, 1182, 1182,
    1219, 1219, 1219, 1219,
    1147, 1147, 1147, 1147,
    18, 18, 18, 18,
    14, 14, 14, 14,
    11, 11, 11, 11,
    20, 20, 20, 20,
]

# ゲームフェーズ定数
PHASE_EARLY = 0   # 序盤（準備フェーズ、サイド差0〜1枚）
PHASE_MID   = 1   # 中盤（積極的攻撃、サイド差2〜4枚）
PHASE_LATE  = 2   # 終盤（詰め、自分か相手のサイド残り1〜2枚）


# ---------------------- 特徴量抽出器 ----------------------

def extract_features(obs: Observation, phase: int = PHASE_MID) -> list[float]:
    """
    約50次元の特徴量ベクトルを盤面から抽出する。
    通常の evaluate_state より大幅に詳細な情報を使用する。
    """
    if obs is None or obs.current is None:
        return [0.0] * 50

    state = obs.current
    yi = state.yourIndex
    oi = 1 - yi
    me = state.players[yi]
    opp = state.players[oi]

    # --- サイド関連 ---
    my_prize_left = len(me.prize)          # 自分の残りサイド枚数
    opp_prize_left = len(opp.prize)        # 相手の残りサイド枚数
    my_prizes_taken = 6 - my_prize_left    # 自分が取ったサイド枚数
    opp_prizes_taken = 6 - opp_prize_left  # 相手が取ったサイド枚数
    prize_diff = my_prizes_taken - opp_prizes_taken  # サイド差（正が有利）

    # --- 自分バトル場の情報 ---
    my_active = me.active[0] if me.active and me.active[0] is not None else None
    my_active_hp = my_active.hp if my_active else 0
    my_active_max_hp = my_active.maxHp if my_active else 0
    my_active_hp_ratio = (my_active_hp / max(1, my_active_max_hp))
    my_active_energy = len(my_active.energies) if my_active else 0
    my_active_id = my_active.id if my_active else 0
    my_active_tools = len(my_active.tools) if my_active else 0
    my_active_preevo = len(my_active.preEvolution) if my_active else 0  # 進化段階

    # 特殊条件ペナルティ
    my_status_penalty = 0.0
    if me.poisoned:  my_status_penalty += 1.0
    if me.burned:    my_status_penalty += 1.5
    if me.asleep:    my_status_penalty += 1.5
    if me.paralyzed: my_status_penalty += 2.0
    if me.confused:  my_status_penalty += 1.0

    # --- 相手バトル場の情報 ---
    opp_active = opp.active[0] if opp.active and opp.active[0] is not None else None
    opp_active_hp = opp_active.hp if opp_active else 100
    opp_active_max_hp = opp_active.maxHp if opp_active else 100
    opp_active_hp_ratio = (opp_active_hp / max(1, opp_active_max_hp))
    opp_active_energy = len(opp_active.energies) if opp_active else 0
    opp_active_id = opp_active.id if opp_active else 0

    # 相手の特殊条件ボーナス
    opp_status_bonus = 0.0
    if opp.poisoned:  opp_status_bonus += 1.0
    if opp.burned:    opp_status_bonus += 1.5
    if opp.asleep:    opp_status_bonus += 1.5
    if opp.paralyzed: opp_status_bonus += 2.0
    if opp.confused:  opp_status_bonus += 1.0

    # --- ベンチ情報 ---
    my_bench_count = len([p for p in me.bench if p is not None])
    opp_bench_count = len([p for p in opp.bench if p is not None])
    my_bench_total_hp = sum(p.hp for p in me.bench if p is not None)
    opp_bench_total_hp = sum(p.hp for p in opp.bench if p is not None)
    my_bench_total_energy = sum(len(p.energies) for p in me.bench if p is not None)
    my_bench_ready = sum(1 for p in me.bench if p is not None and len(p.energies) >= 2)

    # --- 山札・手札情報 ---
    my_deck_count = me.deckCount
    opp_deck_count = opp.deckCount
    my_hand_count = me.handCount if me.hand is None else len(me.hand)
    my_discard_count = len(me.discard)
    opp_discard_count = len(opp.discard)

    # 山切れリスク（残り5枚以下で危険）
    my_deck_risk = max(0.0, 5.0 - my_deck_count) / 5.0
    opp_deck_risk = max(0.0, 5.0 - opp_deck_count) / 5.0

    # --- ゲーム状態 ---
    turn_num = state.turn
    is_supporter_played = float(state.supporterPlayed)
    is_energy_attached = float(state.energyAttached)
    is_retreated = float(state.retreated)
    phase_f = float(phase)

    # --- デッキ固有特徴量（イワパレス軸）---
    # イワパレスがバトル場にいるか
    iwapalace_active = 1.0 if (my_active and my_active.id == 345) else 0.0
    # イワパレスにヒーローマント(1159)が付いているか
    hero_cape_on_iwapalace = 0.0
    # イワパレスにエネルギー3個以上（ジャンボアイス発動条件）
    iwapalace_energy_ge3 = 0.0
    if my_active and my_active.id == 345:
        if any(t.id == 1159 for t in my_active.tools):
            hero_cape_on_iwapalace = 1.0
        if len(my_active.energies) >= 3:
            iwapalace_energy_ge3 = 1.0

    # メガガルーラexがベンチにいるか
    gangaskhan_bench = 1.0 if any(p is not None and p.id == 756 for p in me.bench) else 0.0

    # 相手バトル場の逃げるコスト推定（ボスの指令でロックできるか）
    RETREAT_COSTS = {63: 2, 96: 1, 117: 1, 122: 1, 344: 2, 345: 3, 756: 3}
    opp_retreat_cost = float(RETREAT_COSTS.get(opp_active_id, 1))

    # --- exポケモン判定（きぜつで相手がサイド2枚取る） ---
    EX_IDS = {63, 96, 756}  # タケルライコex, オーガポンex, メガガルーラex
    my_active_is_ex = 1.0 if (my_active and my_active.id in EX_IDS) else 0.0
    opp_active_is_ex = 1.0 if (opp_active and opp_active_id in EX_IDS) else 0.0

    # HPが低い相手がexなら優先的に倒す価値が高い
    opp_ex_low_hp = 1.0 if (opp_active_is_ex and opp_active_hp_ratio < 0.4) else 0.0

    # --- ベンチ空き ---
    my_bench_empty = max(0, me.benchMax - my_bench_count)

    features = [
        # サイド関連（0〜4）
        float(my_prizes_taken),        # 0: 自分取得サイド
        float(opp_prizes_taken),       # 1: 相手取得サイド
        float(prize_diff),             # 2: サイド差
        float(my_prize_left),          # 3: 自分残りサイド
        float(opp_prize_left),         # 4: 相手残りサイド

        # 自分バトル場（5〜11）
        float(my_active_hp),           # 5: 自分バトルHP
        float(my_active_hp_ratio),     # 6: 自分バトルHP割合
        float(my_active_energy),       # 7: 自分バトルエネルギー数
        float(my_active_tools),        # 8: 自分バトルのツール数
        float(my_active_preevo),       # 9: 自分バトルの進化段階
        float(my_status_penalty),      # 10: 自分バトルの状態異常ペナルティ
        float(my_active_is_ex),        # 11: 自分バトルがex

        # 相手バトル場（12〜18）
        float(opp_active_hp),          # 12: 相手バトルHP
        float(opp_active_hp_ratio),    # 13: 相手バトルHP割合
        float(opp_active_energy),      # 14: 相手バトルエネルギー数
        float(opp_status_bonus),       # 15: 相手バトルの状態異常ボーナス
        float(opp_active_is_ex),       # 16: 相手バトルがex
        float(opp_ex_low_hp),          # 17: 相手exが瀕死
        float(opp_retreat_cost),       # 18: 相手逃げるコスト

        # 自分ベンチ（19〜23）
        float(my_bench_count),         # 19: 自分ベンチ数
        float(my_bench_total_hp),      # 20: 自分ベンチHP合計
        float(my_bench_total_energy),  # 21: 自分ベンチエネルギー合計
        float(my_bench_ready),         # 22: 攻撃準備できたベンチポケモン数
        float(my_bench_empty),         # 23: ベンチ空き数

        # 相手ベンチ（24〜25）
        float(opp_bench_count),        # 24: 相手ベンチ数
        float(opp_bench_total_hp),     # 25: 相手ベンチHP合計

        # 山札・手札（26〜32）
        float(my_deck_count),          # 26: 自分山札枚数
        float(opp_deck_count),         # 27: 相手山札枚数
        float(my_hand_count),          # 28: 自分手札枚数
        float(my_discard_count),       # 29: 自分トラッシュ枚数
        float(opp_discard_count),      # 30: 相手トラッシュ枚数
        float(my_deck_risk),           # 31: 自分山切れリスク
        float(opp_deck_risk),          # 32: 相手山切れリスク

        # ゲーム状態（33〜37）
        float(turn_num),               # 33: ターン数
        float(is_supporter_played),    # 34: サポーター使用済み
        float(is_energy_attached),     # 35: エネルギー貼り付け済み
        float(is_retreated),           # 36: にげる使用済み
        float(phase_f),                # 37: ゲームフェーズ

        # デッキ固有（38〜42）
        float(iwapalace_active),       # 38: イワパレスがバトル場
        float(hero_cape_on_iwapalace), # 39: ヒーローマント付きイワパレス
        float(iwapalace_energy_ge3),   # 40: イワパレスにエネルギー3個以上
        float(gangaskhan_bench),       # 41: メガガルーラexがベンチ
        float(opp_retreat_cost),       # 42: 相手逃げるコスト（重複だがフェーズ重み調整用）

        # HP差（43〜44）
        float(my_active_hp - opp_active_hp),    # 43: バトルHP差
        float(my_bench_total_hp - opp_bench_total_hp), # 44: ベンチHP合計差

        # ターン効率（45〜49）
        float(my_prizes_taken) / max(1.0, float(turn_num) / 2),  # 45: ターン当たりサイド取得率
        float(my_hand_count) / max(1.0, float(my_deck_count + my_hand_count)),  # 46: 手札割合
        float(my_bench_count + 1) / 6.0,          # 47: 場のポケモン割合
        1.0 if (my_active and len(my_active.energies) >= 2) else 0.0,  # 48: バトル場が攻撃可能
        float(opp_prizes_taken) / max(1.0, float(turn_num) / 2),  # 49: 相手のターン当たりサイド取得率
    ]
    return features


# ---------------------- 評価関数 ----------------------

# 学習済み重みのデフォルト値（50次元、手動チューニング初期値）
DEFAULT_WEIGHTS_50 = [
    # サイド関連（0〜4）
    150.0,   # 0: 自分取得サイド（勝利に直結）
    -150.0,  # 1: 相手取得サイド
    80.0,    # 2: サイド差
    -30.0,   # 3: 自分残りサイド（少ないほど良い）
    30.0,    # 4: 相手残りサイド

    # 自分バトル場（5〜11）
    0.1,     # 5: 自分バトルHP（絶対値）
    50.0,    # 6: 自分バトルHP割合（重要）
    5.0,     # 7: 自分バトルエネルギー数
    10.0,    # 8: 自分バトルのツール数
    8.0,     # 9: 自分バトルの進化段階
    -15.0,   # 10: 状態異常ペナルティ
    -20.0,   # 11: 自分バトルがex（倒されると2枚取られる）

    # 相手バトル場（12〜18）
    -0.05,   # 12: 相手バトルHP（絶対値、低いほど良い）
    -40.0,   # 13: 相手バトルHP割合（低いほど良い）
    -3.0,    # 14: 相手バトルエネルギー数
    20.0,    # 15: 相手状態異常ボーナス
    30.0,    # 16: 相手バトルがex（倒せると2枚取れる）
    60.0,    # 17: 相手exが瀕死（今すぐ倒せるかも）
    20.0,    # 18: 相手逃げるコスト（ロックしやすい）

    # 自分ベンチ（19〜23）
    8.0,     # 19: 自分ベンチ数
    0.05,    # 20: 自分ベンチHP合計
    3.0,     # 21: 自分ベンチエネルギー合計
    15.0,    # 22: 攻撃準備できたベンチポケモン数
    -5.0,    # 23: ベンチ空き数（少なすぎると次が出せない）

    # 相手ベンチ（24〜25）
    -5.0,    # 24: 相手ベンチ数
    -0.02,   # 25: 相手ベンチHP合計

    # 山札・手札（26〜32）
    0.2,     # 26: 自分山札枚数
    -0.1,    # 27: 相手山札枚数
    2.0,     # 28: 自分手札枚数
    -0.5,    # 29: 自分トラッシュ枚数
    0.3,     # 30: 相手トラッシュ枚数
    -100.0,  # 31: 自分山切れリスク（致命的）
    50.0,    # 32: 相手山切れリスク

    # ゲーム状態（33〜37）
    0.1,     # 33: ターン数
    -5.0,    # 34: サポーター使用済み（使う機会を逃している）
    -3.0,    # 35: エネルギー貼り付け済み
    -2.0,    # 36: にげる使用済み
    0.0,     # 37: ゲームフェーズ（他の特徴量で表現するため0）

    # デッキ固有（38〜42）
    30.0,    # 38: イワパレスがバトル場
    25.0,    # 39: ヒーローマント付きイワパレス
    20.0,    # 40: イワパレスにエネルギー3個以上
    15.0,    # 41: メガガルーラexがベンチ
    15.0,    # 42: 相手逃げるコスト（重複）

    # HP差（43〜44）
    0.05,    # 43: バトルHP差
    0.02,    # 44: ベンチHP合計差

    # ターン効率（45〜49）
    40.0,    # 45: ターン当たりサイド取得率
    5.0,     # 46: 手札割合
    3.0,     # 47: 場のポケモン割合
    10.0,    # 48: バトル場が攻撃可能
    -40.0,   # 49: 相手のターン当たりサイド取得率
]

# 54次元重みのデフォルト値
DEFAULT_WEIGHTS_54 = list(DEFAULT_WEIGHTS_50) + [
    0.01,   # 50: bias_weight (Progressive Bias)
    2.0,    # 51: c_param early phase
    1.414,  # 52: c_param mid phase
    0.8,    # 53: c_param late phase
]

WEIGHTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evolutionary", "best_weights_54.txt"
)


def load_weights_54() -> list[float]:
    """54次元重みをファイルからロードする。なければデフォルト値を返す。"""
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r") as f:
                vals = [float(x.strip()) for x in f.read().split(",") if x.strip()]
            if len(vals) == 54:
                return vals
        except Exception:
            pass
    # 互換性のために50次元の古いファイルがあればロードして拡張する
    old_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "evolutionary", "best_weights_50.txt"
    )
    if os.path.exists(old_file):
        try:
            with open(old_file, "r") as f:
                vals = [float(x.strip()) for x in f.read().split(",") if x.strip()]
            if len(vals) == 50:
                return vals + [0.01, 2.0, 1.414, 0.8]
        except Exception:
            pass
    return list(DEFAULT_WEIGHTS_54)


def save_weights_54(weights: list[float]):
    """54次元重みをファイルに保存する。"""
    try:
        with open(WEIGHTS_FILE, "w") as f:
            f.write(",".join(map(str, weights)))
    except Exception:
        pass


def evaluate_state_50(obs: Observation, weights: list[float], phase: int = PHASE_MID) -> float:
    """50次元特徴量と重みで盤面評価スコアを返す。"""
    features = extract_features(obs, phase)
    return sum(f * w for f, w in zip(features, weights))


# ---------------------- ゲームフェーズ検出 ----------------------

def detect_phase(obs: Observation) -> int:
    """現在のゲームフェーズ（序盤/中盤/終盤）を判定する。"""
    if obs is None or obs.current is None:
        return PHASE_MID

    state = obs.current
    yi = state.yourIndex
    oi = 1 - yi
    my_prize = len(state.players[yi].prize)
    opp_prize = len(state.players[oi].prize)
    turn = state.turn

    # 終盤: どちらかのサイドが2枚以下
    if my_prize <= 2 or opp_prize <= 2:
        return PHASE_LATE

    # 序盤: ターン数が少なく、両者ともサイドを取っていない
    if turn <= 4 and my_prize == 6 and opp_prize == 6:
      # Please note that new elements may be appended to the Enum during the competition.
      return PHASE_EARLY

    return PHASE_MID


# ---------------------- アクション評価ヒューリスティック ----------------------

def score_option(opt) -> float:
    """単一オプションのヒューリスティック評価スコア。"""
    if opt is None:
        return 0.0

    # オプションのタイプに応じて基本優先度を決定
    if opt.type == OptionType.ATTACK:
        return 1000.0  # 攻撃が最も優先される
    elif opt.type == OptionType.ABILITY:
        return 200.0   # 特性は強力
    elif opt.type == OptionType.EVOLVE:
        return 150.0   # 進化
    elif opt.type == OptionType.ATTACH:
        # エネルギーやツールの取り付け
        base = 100.0
        if opt.cardId == 1159:
            base += 50.0
        return base
    elif opt.type == OptionType.PLAY:
        # 手札からのカードプレイ
        base = 50.0
        if opt.cardId in (1147, 1120):  # ネストボールなどの展開札
            base += 30.0
        return base
    elif opt.type == OptionType.RETREAT:
        return 20.0    # 逃げる
    elif opt.type == OptionType.END:
        return 0.0     # ターン終了
    
    return 10.0


def score_action_combination(action_indices: list[int], options) -> float:
    """複数オプションの組み合わせ（アクション）のスコアを算出する。"""
    if not action_indices or not options:
        return 0.0
    
    score = 0.0
    for idx in action_indices:
        if idx < len(options):
            score += score_option(options[idx])
    return score


# ---------------------- 相手デッキ推定 ----------------------

class OpponentDeckTracker:
    """
    ゲームログから相手が使用したカードを追跡し、
    確認済みのカードと未確認のカードを分けてデッキを推定する。
    """
    def __init__(self):
        # 相手がこのゲームで使用したカードIDのリスト（重複あり）
        self.seen_cards: list[int] = []
        self._processed_serials: set[int] = set()

    def update_from_logs(self, logs: list[Log]):
        """ログから相手の使用カードを抽出して追跡リストを更新する。"""
        for log in logs:
            if log.type in (LogType.PLAY, LogType.ATTACH, LogType.MOVE_CARD):
                # 相手（playerIndex != 自分）のカードを記録
                if log.cardId is not None and log.serial is not None:
                    if log.serial not in self._processed_serials:
                        self._processed_serials.add(log.serial)
                        self.seen_cards.append(log.cardId)

    def estimate_deck(self, opp_deck_count: int) -> list[int]:
        """
        確認済みカードをベースに、残り枚数を汎用カードで補完した
        相手デッキのIDリストを返す。
        必ずたねポケモンを含むように保証する。
        """
        if opp_deck_count <= 0:
            return [63]  # 最低1枚は必要

        # 確認済みのカードから始める
        deck = list(self.seen_cards)
        remaining_needed = opp_deck_count - len(deck)

        if remaining_needed <= 0:
            # 多すぎる場合は先頭から切り取る
            result = deck[:opp_deck_count]
        else:
            # 補填用プールをシャッフルして追加
            pool = list(GENERIC_CARD_IDS)
            random.shuffle(pool)
            while remaining_needed > 0 and pool:
                deck.append(pool.pop())
                remaining_needed -= 1
            # さらに足りない場合は草エネルギーで補填
            while len(deck) < opp_deck_count:
                deck.append(18)
            result = deck[:opp_deck_count]

        # たねポケモン（ID 63 = タケルライコex）が含まれているか確認
        # シミュレータはセットアップ時に最低1枚のたねポケモンが必要
        has_basic = any(card_id in STRONG_BASIC_POKEMON_IDS for card_id in result)
        if not has_basic:
            # 最後の1枚をたねポケモンに置き換える
            result[-1] = 63

        return result


# ---------------------- MCTSノード ----------------------

class MctsNode:
    """Progressive Bias対応のMCTSノード。"""
    __slots__ = [
        "search_id", "obs", "parent", "action_from_parent",
        "children", "visit_count", "total_value",
        "untried_actions", "prior_score",
    ]

    def __init__(self, search_id: int, obs: Observation,
                 parent=None, action_from_parent=None, prior_score: float = 0.0):
        self.search_id = search_id
        self.obs = obs
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: list["MctsNode"] = []
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.untried_actions: list = self._get_possible_actions()
        self.prior_score: float = prior_score  # Progressive Bias用の事前スコア

    def _get_possible_actions(self) -> list:
        """選択可能なアクションリストを構築する。"""
        if not self.obs or not self.obs.select:
            return []

        min_c = self.obs.select.minCount
        max_c = self.obs.select.maxCount
        opt_count = len(self.obs.select.option)
        options = self.obs.select.option

        actions = []
        if max_c == 1:
            actions = [[i] for i in range(opt_count)]
        else:
            # 多重選択の場合はサンプリングで候補を絞る
            import itertools
            if opt_count <= 6:
                for r in range(min_c, max_c + 1):
                    for comb in itertools.combinations(range(opt_count), r):
                        actions.append(list(comb))
            else:
                for _ in range(30):
                    cnt = random.randint(min_c, max_c)
                    cand = sorted(random.sample(range(opt_count), cnt))
                    if cand not in actions:
                        actions.append(cand)

        # 探索候補手を評価の昇順でソート（pop()で末尾から取り出されるため）
        actions.sort(key=lambda act: score_action_combination(act, options))
        return actions

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        if not self.obs or not self.obs.current:
            return True
        return self.obs.current.result != -1

    def ucb_score(self, c_param: float, is_my_turn: bool, bias_weight: float = 0.01) -> float:
        """UCB1スコアにProgressive Biasを加えたスコアを返す。"""
        if self.visit_count == 0:
            return float("inf")

        avg_val = self.total_value / self.visit_count
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = c_param * math.sqrt(math.log(max(1, parent_visits)) / self.visit_count)

        # Progressive Bias: 訪問が少ないうちは事前スコアの影響を大きくする
        # prior_score のスケールに合わせてスケーリングします。
        bias = (self.prior_score * bias_weight) / (1.0 + self.visit_count)

        if is_my_turn:
            return avg_val + exploration + bias
        else:
            return -avg_val + exploration + bias

    def best_child(self, c_param: float = 1.414, bias_weight: float = 0.01) -> "MctsNode | None":
        """UCBスコアが最大の子ノードを返す。"""
        if not self.children:
            return None
        is_my_turn = (self.obs.current.yourIndex == 0)
        return max(self.children, key=lambda n: n.ucb_score(c_param, is_my_turn, bias_weight))


# ---------------------- 強化MCTSエージェント ----------------------

class EnhancedMctsAgent(BaseAgent):
    """
    10分の時間制限を活用する強化版MCTSエージェント。
    Progressive Bias、ヒューリスティックプレイアウト、
    ゲームフェーズ対応、相手デッキ推定を統合する。
    """

    # Kaggle環境であっても1試合10分の時間制限があるため、1ターンあたりの思考時間は一律5.0秒に制限します。
    # これによりタイムアウト（TLE）を確実に防ぎます。
    TIME_LIMIT_SECONDS = 5.0


    def __init__(self, deck_path="deck.csv", weights=None, time_limit_override=None, bias_weight=None):
        super().__init__(deck_path)
        all_weights = weights if weights is not None else load_weights_54()
        
        # 54次元重みのうち最初の50次元を盤面評価パラメータとして使用
        self.weights = all_weights[:50]
        
        self.opp_tracker = OpponentDeckTracker()
        
        # 探索パラメータ（Progressive BiasとUCB定数）の設定
        self.bias_weight = bias_weight if bias_weight is not None else all_weights[50]
        self.bias_weight = max(0.0001, self.bias_weight)  # 負値や過小な値を防止
        
        c_early = max(0.01, all_weights[51])
        c_mid   = max(0.01, all_weights[52])
        c_late  = max(0.01, all_weights[53])
        
        self._c_by_phase = {
            PHASE_EARLY: c_early,
            PHASE_MID:   c_mid,
            PHASE_LATE:  c_late,
        }

        # time_limit_override が指定されていればそちらを優先する（ベンチマーク用）
        if time_limit_override is not None:
            self._time_limit = float(time_limit_override)
        else:
            self._time_limit = self.TIME_LIMIT_SECONDS

        # プレイアウトの深さ: フェーズで調整
        self._playout_depth_by_phase = {
            PHASE_EARLY: 8,
            PHASE_MID:   5,
            PHASE_LATE:  3,
        }


    def _select_action_impl(self, obs: Observation) -> list[int]:
        start_time = time.time()

        # ログから相手カードを追跡
        if obs.logs:
            self.opp_tracker.update_from_logs(obs.logs)

        # ゲームフェーズを検出
        phase = detect_phase(obs)
        c_param = self._c_by_phase[phase]
        playout_depth = self._playout_depth_by_phase[phase]

        try:
            # デッキ情報の構築
            my_deck = self._predict_my_deck(obs)
            my_prize = self._predict_my_prize(obs)

            opp_idx = 1 - obs.current.yourIndex
            opp_player = obs.current.players[opp_idx]
            opp_deck = self.opp_tracker.estimate_deck(opp_player.deckCount)
            opp_prize = [1] * len(opp_player.prize)
            opp_hand = [18] * opp_player.handCount  # 汎用カードで補填

            # 相手バトル場が裏向きの場合
            opp_active = []
            if opp_player.active and opp_player.active[0] is None:
                opp_active = [63]  # タケルライコexとして仮定（最も多いたね）

            # 探索ルート状態の初期化
            root_state = search_begin(
                agent_observation=obs,
                your_deck=my_deck,
                your_prize=my_prize,
                opponent_deck=opp_deck,
                opponent_prize=opp_prize,
                opponent_hand=opp_hand,
                opponent_active=opp_active,
            )

            root_node = MctsNode(root_state.searchId, obs)

            # 選択可能なアクションが1つしかない場合は、探索せずに即座に決定（0秒）
            if len(root_node.untried_actions) == 1:
                search_end()
                return root_node.untried_actions[0]
            elif len(root_node.untried_actions) == 0:
                search_end()
                return []

            # 時間制限まで反復してMCTSを実行
            iteration_count = 0
            while time.time() - start_time < self._time_limit:
                node = root_node
                path_to_release = []

                # SELECT
                while not node.is_terminal() and node.is_fully_expanded() and node.children:
                    next_node = node.best_child(c_param, self.bias_weight)
                    if next_node is None:
                        break
                    node = next_node

                # EXPAND
                if not node.is_terminal() and not node.is_fully_expanded():
                    action = node.untried_actions.pop()
                    try:
                        next_state = search_step(node.search_id, action)
                        next_obs = next_state.observation

                        # 事前スコア（Progressive Bias）を計算
                        prior = evaluate_state_50(next_obs, self.weights, phase)

                        child = MctsNode(
                            next_state.searchId, next_obs,
                            parent=node, action_from_parent=action,
                            prior_score=prior,
                        )
                        node.children.append(child)
                        node = child
                    except Exception:
                        continue

                # SIMULATE (ヒューリスティックプレイアウト)
                val = self._heuristic_playout(node, path_to_release, playout_depth, phase)

                # プレイアウト中の一時状態を解放
                for tid in path_to_release:
                    try:
                        search_release(tid)
                    except Exception:
                        pass

                # BACKPROPAGATE
                cur = node
                while cur is not None:
                    cur.visit_count += 1
                    cur.total_value += val
                    cur = cur.parent

                iteration_count += 1

                # 早期終了チェック（50イテレーションごとに実行）
                if iteration_count % 50 == 0:
                    elapsed_check = time.time() - start_time
                    if elapsed_check > 1.0 and root_node.children:
                        sorted_children = sorted(root_node.children, key=lambda c: c.visit_count, reverse=True)
                        v1 = sorted_children[0].visit_count
                        v2 = sorted_children[1].visit_count if len(sorted_children) > 1 else 0
                        
                        avg_time = elapsed_check / iteration_count
                        remaining_time = self._time_limit - elapsed_check
                        max_remaining_iters = remaining_time / max(1e-5, avg_time)
                        
                        # 安全マージンとして残りイテレーション数を 1.05倍 で計算
                        if v1 > v2 + max_remaining_iters * 1.05:
                            break

            elapsed = time.time() - start_time
            print(f"MCTS: {iteration_count}イテレーション完了 ({elapsed:.1f}秒, フェーズ={phase})")

            # 最多訪問ノードのアクションを選択
            best_action = None
            if root_node.children:
                best_child = max(root_node.children, key=lambda c: c.visit_count)
                best_action = best_child.action_from_parent

            search_end()

            if best_action is not None:
                return best_action

        except Exception as e:
            print(f"強化MCTS: エラーが発生したためランダムにフォールバックします: {e}")
            try:
                search_end()
            except Exception:
                pass

        # フォールバック
        opt_count = len(obs.select.option)
        cnt = obs.select.maxCount
        return random.sample(range(opt_count), cnt)

    def _heuristic_playout(
        self, node: MctsNode, path_to_release: list,
        max_depth: int, phase: int
    ) -> float:
        """
        超軽量ヒューリスティックを導入した確率的プレイアウト。
        """
        if node.is_terminal():
            res = node.obs.current.result
            if res == 0:
                return 10000.0
            elif res == 1:
                return -10000.0
            return 0.0

        current_id = node.search_id
        current_obs = node.obs

        for depth in range(max_depth):
            if not current_obs or not current_obs.select:
                break
            if current_obs.current.result != -1:
                break

            min_c = current_obs.select.minCount
            max_c = current_obs.select.maxCount
            opt_count = len(current_obs.select.option)
            options = current_obs.select.option

            # 75%の確率でヒューリスティックに最適な手を選択、25%は多様性のためにランダム
            if random.random() < 0.75:
                if max_c == 1:
                    best_opt_idx = max(range(opt_count), key=lambda i: score_option(options[i]))
                    action = [best_opt_idx]
                else:
                    count = random.randint(min_c, max_c)
                    sorted_opt_indices = sorted(
                        range(opt_count),
                        key=lambda i: score_option(options[i]),
                        reverse=True
                    )
                    action = sorted(sorted_opt_indices[:count])
            else:
                count = random.randint(min_c, max_c)
                action = random.sample(range(opt_count), count)
                action.sort()

            try:
                next_state = search_step(current_id, action)
                current_id = next_state.searchId
                current_obs = next_state.observation
                path_to_release.append(current_id)
            except Exception:
                break

        if current_obs:
            if current_obs.current.result == 0:
                return 10000.0
            elif current_obs.current.result == 1:
                return -10000.0
            return evaluate_state_50(current_obs, self.weights, phase)
        return 0.0

    def _predict_my_deck(self, obs: Observation) -> list[int]:
        """公開情報から自分の山札に残るカードを推定する。"""
        deck = self.read_deck_csv()
        yi = obs.current.yourIndex
        me = obs.current.players[yi]

        def remove_card(card_id):
            if card_id in deck:
                deck.remove(card_id)

        for c in me.discard:
            remove_card(c.id)
        if me.hand:
            for c in me.hand:
                remove_card(c.id)

        def remove_pokemon(p):
            if p is None:
                return
            remove_card(p.id)
            for c in p.energyCards:
                remove_card(c.id)
            for c in p.tools:
                remove_card(c.id)
            for c in p.preEvolution:
                remove_card(c.id)

        for p in me.active:
            remove_pokemon(p)
        for p in me.bench:
            remove_pokemon(p)

        while len(deck) > me.deckCount and deck:
            deck.pop()
        while len(deck) < me.deckCount:
            deck.append(1)

        return deck

    def _predict_my_prize(self, obs: Observation) -> list[int]:
        """サイド枚数に合わせてダミーを返す。"""
        yi = obs.current.yourIndex
        return [1] * len(obs.current.players[yi].prize)
