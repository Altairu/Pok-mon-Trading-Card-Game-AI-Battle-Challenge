from .base_agent import BaseAgent
from .random_agent import RandomAgent
from .evolutionary.evolutionary_agent import EvolutionaryAgent
from .mcts_agent import MctsAgent
from .rl_agent import RlAgent
from .pytorch_rl_agent import PytorchRlAgent
from .enhanced_mcts_agent import EnhancedMctsAgent

def get_agent(agent_type: str = "random", **kwargs) -> BaseAgent:
    """
    指定されたタイプのエージェントインスタンスを作成して返します。
    これにより、アルゴリズムやアプローチの変更が容易になります。
    """
    if agent_type == "random":
        return RandomAgent(**kwargs)
    elif agent_type == "evolutionary":
        return EvolutionaryAgent(**kwargs)
    elif agent_type == "mcts":
        return MctsAgent(**kwargs)
    elif agent_type == "rl":
        return RlAgent(**kwargs)
    elif agent_type == "pytorch_rl":
        return PytorchRlAgent(**kwargs)
    elif agent_type == "enhanced_mcts":
        return EnhancedMctsAgent(**kwargs)
    else:
        raise ValueError(f"未知のエージェントタイプ: {agent_type}")
