import sys
import os
import webbrowser

# プロジェクトルートのパスを追加してインポート可能にします
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evolutionary.dashboard import run_server

if __name__ == "__main__":
    print("==================================================")
    print(" ポケカABC - 進化計算（GA）トレーニングダッシュボード")
    print("==================================================")
    
    port = 5000
    url = f"http://127.0.0.1:{port}"
    
    print(f"\nWebサーバーを起動します。ブラウザで以下を開いて学習を開始してください:")
    print(f" => {url}")
    print("\n※ トレーニングサーバーを停止するには、ターミナルで Ctrl+C を押してください。")
    print("==================================================\n")
    
    # ブラウザを自動的に開きます
    try:
        webbrowser.open(url)
    except Exception:
        pass
        
    # Flask サーバーを起動します
    run_server(port=port)
