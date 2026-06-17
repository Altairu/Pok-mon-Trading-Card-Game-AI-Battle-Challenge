# ポケモンカードAIコンペ (ポケカABC) 開発環境

本リポジトリは、Pokémon TCG AI Battle Challenge に参加するためのローカル開発環境、ゲームエンジン接続モジュール（cgパッケージ）、および開発用エージェントの構成テンプレートです。

## プロジェクト構成

* main.py: エージェントのエントリーポイント（Kaggle提出必須ファイル）
* deck.csv: 60枚のデッキリスト（Kaggle提出必須ファイル）
* requirements.txt: ローカル開発環境に必要なパッケージの定義ファイル
* cg/: ゲームエンジンとやり取りするためのPython APIおよびネイティブライブラリ
* src/: 開発用ソースコードディレクトリ
  * base_agent.py: エージェント共通の基底クラス
  * random_agent.py: ランダムにアクションを選択するベースライン
  * agent_factory.py: エージェントの生成と切り替えを行うファクトリ
  * mcts_agent.py: モンテカルロ木探索（MCTS）をベースにしたエージェント
  * rl_agent.py: 時間的差分（TD）学習を用いた線形評価モデルエージェント
  * pytorch_rl_agent.py: PyTorchで定義された価値ネットワークを使用した強化学習エージェント
  * evolutionary/: 進化計算およびダッシュボード開発用ディレクトリ
    * evolutionary_agent.py: 進化計算（GA）エージェントの実装
    * ga.py: 遺伝的アルゴリズムおよび強化学習のWeb動作用バックエンド処理
    * dashboard.py: Flaskを使用した統合Webサーバー
    * templates/ / static/: ダッシュボードおよびビジュアライザのフロントエンド画面
* scripts/build.py: 提出用の submission.tar.gz を生成するビルドスクリプト
* scripts/test_agent.py: エージェントが正常に動作するかを検証するテストスクリプト
* scripts/train_ga.py: ローカルのコンソールで遺伝的アルゴリズムを実行するスクリプト
* scripts/train_pytorch_rl.py: ローカルのコンソールでPyTorch強化学習の自己対戦トレーニングを実行するスクリプト
* scripts/start_dashboard.py: 統合トレーニングダッシュボードを起動するスクリプト

## 環境構築手順

ローカルのPython仮想環境（venv）を構築し、必要なパッケージをインストールします。

Windows (PowerShell) の場合：

* 仮想環境を作成します
  ```powershell
  python -m venv venv
  ```
* 仮想環境を有効化します
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* 依存関係をインストールします
  ```powershell
  pip install -r requirements.txt
  ```

macOS / Linux の場合：

* 仮想環境を作成します
  ```bash
  python3 -m venv venv
  ```
* 仮想環境を有効化します
  ```bash
  source venv/bin/activate
  ```
* 依存関係をインストールします
  ```bash
  source pip install -r requirements.txt
  ```

## 動作確認手順

作成したエージェントがエラーなく動作するかローカルで確認します。

* テストスクリプトを実行します
  ```bash
  python scripts/test_agent.py
  ```

## 統合Webダッシュボードの使用方法

以下のコマンドを実行することで、Webブラウザから強化学習（PyTorch）や遺伝的アルゴリズム（GA）のトレーニングの開始や、対戦の可視化を行えるダッシュボードが起動します。

```bash
python scripts/start_dashboard.py
```

実行するとWebサーバーが起動し、自動的にブラウザで以下のURLが開きます。
`http://127.0.0.1:5000/`

* トレーニング画面
  「学習アルゴリズム」から「遺伝的アルゴリズム (GA)」または「PyTorch 強化学習 (Self-Play)」を選択し、パラメータを設定して「学習開始」を押すことでトレーニングを開始できます。
  進捗状況が画面上にリアルタイムで表示され、評価値やテスト勝率の推移がグラフに描画されます。

* 対戦ビジュアライザ画面
  「対戦ビジュアライザ」のタブから、開発した「PyTorch強化学習 (pytorch_rl)」などの各エージェント同士をシミュレーション対戦させ、実際の対戦中の手札や盤面の状況を可視化してコマ送りや自動再生で観戦できます。対戦結果のバナーには、具体的な敗因（山札切れなど）が表示されます。

## コマンドラインでのトレーニング実行

Webブラウザを使用せず、コマンドライン上で直接PyTorch強化学習の自己対戦を回してモデルを育てることも可能です。

```bash
python scripts/train_pytorch_rl.py
```
* 実行中のモデルパラメータは `src/evolutionary/pytorch_model.pth` に随時上書き保存されます。
* 500ゲームごとの評価テストで過去最高勝率を更新すると、ベストモデルが `src/evolutionary/pytorch_model_best.pth` に自動保存されます。
* 途中で学習を終了したい場合は、ターミナルで Ctrl+C を入力することで、その時点でのモデルを保存して終了できます。

## 提出ファイルの作成手順

Kaggleにアップロードするための提出用アーカイブ（submission.tar.gz）を作成します。

* ビルドスクリプトを実行します
  ```bash
  python scripts/build.py
  ```
* 実行後、プロジェクトのルートディレクトリに submission.tar.gz が生成されます。このファイルをそのままKaggleの提出ページへアップロードしてください。
