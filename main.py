from src.agent_factory import get_agent

# エージェントインスタンスのキャッシュ用変数
_agent_instance = None

def agent(obs_dict: dict) -> list[int]:
    """
    ポケモンカードAIコンペ（ポケカABC）のエントリポイント。
    """
    global _agent_instance
    if _agent_instance is None:
        # 強化MCTSエージェントを使用する。
        # 10分の時間制限を活用し、Progressive Bias付きMCTS＋ヒューリスティックプレイアウトで
        # 高精度な意思決定を行う。
        _agent_instance = get_agent("enhanced_mcts")
        # "enhanced_mcts" Progressive Bias、ヒューリスティックプレイアウト、フェーズ対応の強化MCTSエージェント
        # "mcts"         モンテカルロ木探索（旧バージョン、50イテレーション固定）
        # "rl"           時間的差分学習による線形評価モデルエージェント
        # "pytorch_rl"   PyTorchニューラルネットワークを使用した強化学習エージェント
        # "evolutionary" 遺伝的アルゴリズムで最適化されたルールベースエージェント
        # "random"       ランダムにアクションを選択するベースライン
    return _agent_instance.select_action(obs_dict)
