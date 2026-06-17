from src.agent_factory import get_agent

# エージェントインスタンスのキャッシュ用変数
_agent_instance = None

def agent(obs_dict: dict) -> list[int]:
    """
    ポケモンカードAIコンペ（ポケカABC）のエントリポイント。
    """
    global _agent_instance
    if _agent_instance is None:
        # 遺伝的アルゴリズムを使用する進化計算エージェントをロードします。
        # 動作テストや元のベースラインに戻す場合は、引数を "random" に変更します。
        _agent_instance = get_agent("evolutionary")
        
    return _agent_instance.select_action(obs_dict)
