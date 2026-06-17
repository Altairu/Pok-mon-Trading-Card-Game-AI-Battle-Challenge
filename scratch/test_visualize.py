import sys
import os

# プロジェクトルートのパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cg.game import battle_start, battle_select, battle_finish, visualize_data
from src.agent_factory import get_agent

def main():
    agent_me = get_agent("random")
    agent_opp = get_agent("random")
    
    deck0 = agent_me.read_deck_csv()
    deck1 = agent_opp.read_deck_csv()
    
    obs_dict, start_data = battle_start(deck0, deck1)
    
    print("=== Start Data ===")
    print(start_data)
    
    print("=== Visualize Data ===")
    vis = visualize_data()
    print(vis[:1000])  # 最初の1000文字を出力
    
    # 1ターン進めてみる
    if obs_dict and obs_dict.get("current"):
        your_idx = obs_dict["current"]["yourIndex"]
        action = agent_me.select_action(obs_dict) if your_idx == 0 else agent_opp.select_action(obs_dict)
        obs_dict = battle_select(action)
        print("=== After 1 Action ===")
        vis_after = visualize_data()
        print(vis_after[:1000])

    battle_finish()

if __name__ == "__main__":
    main()
