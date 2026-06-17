import json
import os
import sys

# Configure stdout to use utf-8
sys.stdout.reconfigure(encoding='utf-8')

project_root = r"c:\Users\106no\Documents\GitHub\Pokémon Trading Card Game AI Battle Challenge"
sys.path.append(project_root)

from src.agent_factory import get_agent
from cg.game import battle_start, battle_select, battle_finish

def run_test_match(p0_name, p1_name):
    print(f"\n--- Testing Match: {p0_name} vs {p1_name} ---")
    
    # Load default dummy weights
    from src.evolutionary.ga import DEFAULT_WEIGHTS
    weights = list(DEFAULT_WEIGHTS)
    
    try:
        use_weight_agents = ["evolutionary", "mcts", "rl"]
        p0_agent = get_agent(p0_name, weights=weights) if p0_name in use_weight_agents else get_agent(p0_name)
        p1_agent = get_agent(p1_name, weights=weights) if p1_name in use_weight_agents else get_agent(p1_name)
        
        deck0 = p0_agent.read_deck_csv()
        deck1 = p1_agent.read_deck_csv()
        
        obs_dict, start_data = battle_start(deck0, deck1)
        if not obs_dict:
            print("Failed to start battle.")
            return False
            
        turn = 0
        while turn < 150: # Limit turns for fast test
            current_state = obs_dict.get("current")
            if current_state is not None:
                result = current_state.get("result", -1)
                if result != -1:
                    print(f"Match finished at turn {turn}. Winner index: {result}")
                    break
                    
            your_idx = obs_dict.get("current", {}).get("yourIndex", 0)
            if your_idx == 0:
                action = p0_agent.select_action(obs_dict)
            else:
                action = p1_agent.select_action(obs_dict)
                
            obs_dict = battle_select(action)
            turn += 1
            
        battle_finish()
        print(f"Test match successfully completed in {turn} turns.")
        return True
        
    except Exception as e:
        print(f"Error during test match: {e}")
        try:
            battle_finish()
        except:
            pass
        return False

# Test MCTS and RL agents
success_mcts = run_test_match("mcts", "random")
success_rl = run_test_match("rl", "random")

if success_mcts and success_rl:
    print("\nAll agent tests passed successfully!")
    sys.exit(0)
else:
    print("\nSome tests failed.")
    sys.exit(1)
