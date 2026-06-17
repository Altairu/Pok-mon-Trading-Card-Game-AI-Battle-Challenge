import sys
import os
import webbrowser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evolutionary.dashboard import run_server

if __name__ == "__main__":
    print("==================================================")
    print(" ポケカABC - 統合トレーニングダッシュボード")
    print("==================================================")
    
    port = 5000
    url = f"http://127.0.0.1:{port}"
    
    print(f"\nWebサーバーを起動します。ブラウザで以下を開いて学習やシミュレーションを開始してください:")
    print(f" => {url}")
    print("\n※ サーバーを停止するには、ターミナルで Ctrl+C を押してください。")
    print("==================================================\n")
    
    try:
        webbrowser.open(url)
    except Exception:
        pass
        
    run_server(port=port)
