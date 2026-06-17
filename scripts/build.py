import os
import tarfile

def build_submission():
    """
    提出用の submission.tar.gz を作成するスクリプト。
    main.py と deck.csv をアーカイブのルート直下に配置します。
    """
    archive_name = "submission.tar.gz"
    
    # アーカイブに含めるファイルのリスト
    files_to_include = [
        "main.py",
        "deck.csv",
        "cg",
        "src"
    ]
    
    # スクリプトの配置場所からプロジェクトのルートディレクトリを特定します
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    archive_path = os.path.join(project_root, archive_name)
    
    print("提出用アーカイブ (submission.tar.gz) の作成を開始します...")
    
    # 既存の古いアーカイブを削除します
    if os.path.exists(archive_path):
        os.remove(archive_path)
        print(f"既存の {archive_name} を削除しました。")
        
    # tar.gz ファイルを作成します
    with tarfile.open(archive_path, "w:gz") as tar:
        for file_name in files_to_include:
            file_path = os.path.join(project_root, file_name)
            if os.path.exists(file_path):
                # arcname を指定して、アーカイブ内のルートに配置します
                tar.add(file_path, arcname=file_name)
                print(f"追加: {file_name}")
            else:
                print(f"エラー: 必須ファイル {file_name} が見つかりません。")
                return
                
    print(f"作成が完了しました: {archive_name}")

if __name__ == "__main__":
    build_submission()
