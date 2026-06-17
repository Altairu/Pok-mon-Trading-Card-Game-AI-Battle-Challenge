import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cg.game import battle_start, battle_select, battle_finish, visualize_data
from src.agent_factory import get_agent

def main():
    agent_me = get_agent("random")
    agent_opp = get_agent("random")
    
    deck0 = agent_me.read_deck_csv()
    deck1 = agent_opp.read_deck_csv()
    
    obs_dict, start_data = battle_start(deck0, deck1)
    
    print("Initial list length:", len(json.loads(visualize_data())))
    
    for i in range(10):
        current_state = obs_dict.get("current")
        if current_state is not None and current_state.get("result", -1) != -1:
            print("Game finished")
            break
            
        your_idx = obs_dict.get("current", {}).get("yourIndex", 0)
        action = agent_me.select_action(obs_dict) if your_idx == 0 else agent_opp.select_action(obs_dict)
        obs_dict = battle_select(action)
        
        vis_list = json.loads(visualize_data())
        print(f"Step {i+1} list length:", len(vis_list))
        
    battle_finish()

if __name__ == "__main__":
    main()
