import sys
import os

# プロジェクトルートのパスを追加して main.py をインポートできるようにします
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import agent

# cg パッケージの読み込みテスト
try:
    from cg.api import Observation, to_observation_class
    print("cg パッケージのインポートに成功しました。")
except ImportError as e:
    print(f"エラー: cg パッケージのインポートに失敗しました。 {e}")
    sys.exit(1)

def test():
    print("ゲームエンジン (cgパッケージ) 統合後のエージェントの動作テストを開始します。")
    
    # テスト1: 初期選択時 (obs に select 情報が含まれない場合)
    # 60枚のデッキリスト（カードIDのリスト）が返されることを期待します
    obs_initial = {
        "select": None,
        "logs": [],
        "current": None
    }
    action_initial = agent(obs_initial)
    print(f"テスト1 (初期デッキ選択) 出力枚数: {len(action_initial)}")
    assert isinstance(action_initial, list), "テスト1失敗: 返り値がリストではありません"
    assert len(action_initial) == 60, f"テスト1失敗: デッキの枚数が60枚ではありません (現在: {len(action_initial)}枚)"
    for card_id in action_initial:
        assert isinstance(card_id, int), f"テスト1失敗: デッキに含まれる値が整数ではありません ({card_id})"
    print("テスト1 成功: 60枚のデッキリストが正しく取得できました。")
    
    # テスト2: 通常ターン時 (obs.select に選択肢が含まれる場合)
    # 合法手の中から minCount〜maxCount 個のアクションインデックスが返されることを期待します
    obs_turn = {
        "select": {
            "type": 0,  # SelectType.MAIN
            "context": 0,  # SelectContext.MAIN
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": [
                {"type": 13, "attackId": 1},  # OptionType.ATTACK
                {"type": 14}  # OptionType.END
            ],
            "deck": None,
            "contextCard": None,
            "effect": None
        },
        "logs": [],
        "current": {
            "turn": 1,
            "turnActionCount": 0,
            "yourIndex": 0,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "looking": None,
            "players": [
                {
                    "active": [],
                    "bench": [],
                    "benchMax": 5,
                    "deckCount": 40,
                    "discard": [],
                    "prize": [None, None, None, None, None, None],
                    "handCount": 7,
                    "hand": [],
                    "poisoned": False,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False
                },
                {
                    "active": [],
                    "bench": [],
                    "benchMax": 5,
                    "deckCount": 40,
                    "discard": [],
                    "prize": [None, None, None, None, None, None],
                    "handCount": 7,
                    "hand": None,
                    "poisoned": False,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False
                }
            ]
        }
    }
    action_turn = agent(obs_turn)
    print(f"テスト2 (通常ターン選択) 出力アクション: {action_turn}")
    assert isinstance(action_turn, list), "テスト2失敗: 返り値がリストではありません"
    assert len(action_turn) == 1, f"テスト2失敗: 選択されたアクション数が期待値 1 と異なります (現在: {len(action_turn)})"
    assert action_turn[0] in [0, 1], f"テスト2失敗: アクションのインデックス範囲が不正です ({action_turn[0]})"
    print("テスト2 成功: ターン中のランダムアクション選択が正常に動作しました。")
    
    print("すべてのローカル動作テストが正常に通過しました。")

if __name__ == "__main__":
    test()
