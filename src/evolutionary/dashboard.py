import threading
import json
import csv
import os
from flask import Flask, jsonify, request, render_template
from .ga import training_status, train_ga_loop

app = Flask(__name__, template_folder='templates', static_folder='static')
status_lock = threading.Lock()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    with status_lock:
        return jsonify(training_status)

@app.route("/api/start", methods=["POST"])
def start_training():
    with status_lock:
        if training_status["running"]:
            return jsonify({"status": "error", "message": "既に学習が実行中です。"})
            
        data = request.json or {}
        
        # UIから送られたトレーニングパラメータを反映します
        training_status["algorithm"] = data.get("algorithm", "ga")
        training_status["total_generations"] = int(data.get("generations", 50))
        training_status["population_size"] = int(data.get("population_size", 20))
        training_status["num_games_per_eval"] = int(data.get("num_games", 3))
        training_status["timer_limit"] = int(data.get("timer_limit", 10800))  # 秒単位
        
        training_status["stop_requested"] = False
        training_status["pause_requested"] = False
        training_status["paused"] = False
        
        # 学習ループを別スレッドで開始
        if training_status["algorithm"] == "pytorch_rl":
            from .ga import train_pytorch_rl_loop
            thread = threading.Thread(target=train_pytorch_rl_loop)
        else:
            thread = threading.Thread(target=train_ga_loop)
            
        thread.daemon = True
        thread.start()
        
    return jsonify({"status": "success", "message": "学習を開始しました。"})

@app.route("/api/stop", methods=["POST"])
def stop_training():
    with status_lock:
        if not training_status["running"]:
            return jsonify({"status": "error", "message": "学習は実行されていません。"})
        training_status["stop_requested"] = True
        training_status["pause_requested"] = False
        training_status["paused"] = False
        training_status["message"] = "停止リクエスト送信済み"
    return jsonify({"status": "success", "message": "停止処理を開始しました。"})

@app.route("/api/pause", methods=["POST"])
def pause_training():
    with status_lock:
        if not training_status["running"]:
            return jsonify({"status": "error", "message": "学習は実行されていません。"})
            
        data = request.json or {}
        pause_flag = bool(data.get("pause", False))
        training_status["pause_requested"] = pause_flag
        training_status["paused"] = pause_flag
        training_status["message"] = "一時停止中" if pause_flag else "再開中..."
    return jsonify({"status": "success", "message": "一時停止状態を変更しました。"})

@app.route("/visualizer")
def visualizer():
    return render_template("visualizer.html")

@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.json or {}
    p0_name = data.get("agent_p0", "evolutionary")
    p1_name = data.get("agent_p1", "random")
    
    from .ga import load_best_weights
    weights = load_best_weights()
    
    try:
        from src.agent_factory import get_agent
        from cg.game import battle_start, battle_select, battle_finish, visualize_data
        
        use_weight_agents = ["evolutionary", "mcts", "rl"]
        p0_agent = get_agent(p0_name, weights=weights) if p0_name in use_weight_agents else get_agent(p0_name)
        p1_agent = get_agent(p1_name, weights=weights) if p1_name in use_weight_agents else get_agent(p1_name)
        
        deck0 = p0_agent.read_deck_csv()
        deck1 = p1_agent.read_deck_csv()
        
        obs_dict, start_data = battle_start(deck0, deck1)
        if not obs_dict:
            return jsonify({"status": "error", "message": "対戦の開始に失敗しました。"})
            
        turn = 0
        while turn < 1000:
            current_state = obs_dict.get("current")
            if current_state is not None:
                result = current_state.get("result", -1)
                if result != -1:
                    break
                    
            your_idx = obs_dict.get("current", {}).get("yourIndex", 0)
            if your_idx == 0:
                action = p0_agent.select_action(obs_dict)
            else:
                action = p1_agent.select_action(obs_dict)
                
            obs_dict = battle_select(action)
            turn += 1
            
        vis_json_str = visualize_data()
        battle_finish()
        
        vis_data = json.loads(vis_json_str)
        return jsonify({"status": "success", "data": vis_data})
        
    except Exception as e:
        try:
            battle_finish()
        except:
            pass
        return jsonify({"status": "error", "message": f"シミュレーション中にエラーが発生しました: {str(e)}"})

_jp_card_data_cache = None

def load_jp_card_data():
    global _jp_card_data_cache
    if _jp_card_data_cache is not None:
        return _jp_card_data_cache
        
    _jp_card_data_cache = {}
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "pokemon-tcg-ai-battle", "JP_Card_Data.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join("pokemon-tcg-ai-battle", "JP_Card_Data.csv")
        
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if not row or len(row) < 17:
                        continue
                    card_id = int(row[0])
                    card_name = row[1]
                    evolves_from = row[7] if row[7] != "n/a" else None
                    attack_name = row[13] if row[13] != "n/a" else None
                    attack_dmg = row[15] if row[15] != "n/a" else None
                    effect_text = row[16] if row[16] != "n/a" else None
                    
                    if card_id not in _jp_card_data_cache:
                        _jp_card_data_cache[card_id] = {
                            "name": card_name,
                            "evolvesFrom": evolves_from,
                            "attacks": [],
                            "skills": []
                        }
                    
                    if attack_name:
                        _jp_card_data_cache[card_id]["attacks"].append({
                            "name": attack_name,
                            "damage": attack_dmg,
                            "text": effect_text
                        })
                    elif effect_text:
                        _jp_card_data_cache[card_id]["skills"].append({
                            "name": card_name,
                            "text": effect_text
                        })
        except Exception as e:
            print(f"JP_Card_Data.csv のロード中にエラーが発生しました: {e}")
            
    return _jp_card_data_cache

@app.route("/api/cards", methods=["GET"])
def get_cards():
    try:
        from cg.api import all_card_data
        from dataclasses import asdict
        cards = all_card_data()
        cards_dict = [asdict(c) for c in cards]
        
        jp_data = load_jp_card_data()
        
        for c in cards_dict:
            cid = c["cardId"]
            if cid in jp_data:
                c["name"] = jp_data[cid]["name"]
                if jp_data[cid]["evolvesFrom"]:
                    c["evolvesFrom"] = jp_data[cid]["evolvesFrom"]
                
                if c["cardType"] != 0 and jp_data[cid]["skills"]:
                    c["skills"] = jp_data[cid]["skills"]
                    
        return jsonify({"status": "success", "cards": cards_dict})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/api/attacks", methods=["GET"])
def get_attacks():
    try:
        from cg.api import all_attack, all_card_data
        from dataclasses import asdict
        attacks = all_attack()
        attacks_dict = [asdict(a) for a in attacks]
        
        cards = all_card_data()
        jp_data = load_jp_card_data()
        
        jp_attacks_map = {}
        for c in cards:
            cid = c.cardId
            if cid in jp_data:
                csv_attacks = jp_data[cid].get("attacks", [])
                api_attack_ids = c.attacks
                
                for idx, a_id in enumerate(api_attack_ids):
                    if idx < len(csv_attacks):
                        jp_attacks_map[a_id] = csv_attacks[idx]
                        
        for a in attacks_dict:
            aid = a["attackId"]
            if aid in jp_attacks_map:
                a["name"] = jp_attacks_map[aid]["name"]
                if jp_attacks_map[aid]["text"]:
                    a["text"] = jp_attacks_map[aid]["text"]
                    
        return jsonify({"status": "success", "attacks": attacks_dict})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def run_server(port=5000):
    """Flaskサーバーを起動します。"""
    app.run(host="127.0.0.1", port=port, debug=False)
