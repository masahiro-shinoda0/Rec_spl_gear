# recboleを用いた推薦システムの作成
[`RecBole`](https://www.recbole.io/index.html)を使ったゼミの課題に取り組むうえで，気になったことを以下に書き記す．\
2025/12/08に作成．
2025/12/22から，[`stat.ink`](https://stat.ink/)のデータを利用した推薦システムの作成を開始．\
2026/01/06から，[`Google Colab`](https://colab.research.google.com/)上で動く[`Streamlit`](https://streamlit.io/)を利用したUIの作成を開始．\
2026/01/18から，[`FastAPI`](https://fastapi.tiangolo.com/)をバックエンドに使用したWebアプリケーションの作成を開始．


以下は`Git`の使い方をまとめたものである．\
[Gitの使い方まとめ](https://github.com/masahiro-shinoda0/AtCoder/blob/main/readme/HowToUseGitHub.md)


## 2025/12/08
`conda`を使用するために`Anaconda`を使用する．
`Anaconda Prompt` を起動し，以下のコマンドを実行．
```
conda activate recbole
```
これは自前ノートパソコンで`RecBole`を使用するために用いたためであり，以降ではサーバーに仮想環境を構築して行ったため，`Anaconda`は使用していない．

## RecBoleについて
`RecBole`とは，`PyTorch`ベースの，推薦アルゴリズムのライブラリである．推薦アルゴリズムのライブラリは，他に`Recommenders`や`Spotlight`などがある．\
[RecBoleの公式ドキュメント](https://www.recbole.io/docs/)を参照して，`Parameter Tuning`等を行う．

### UNIX/Linuxコマンド一覧
`Linux`のコマンドをメモする．以下，コマンドをまとめた表である．まずは，基本的なもの．
- 基本的なコマンド
  - `ls`で現在のディレクトリを表示．カレントディレクトリ
  - `cd`でファイルのある階層に移動
  - `python xxx.py`で`Python`を実行
  - `pwd`で現在の階層を表示する
  - `dir`でファイルを一覧表示

- ファイル，フォルダ操作
  - `touch`でファイルを作成
  - `ren`または`rename`でファイル名を変更
  - `cp` <コピー元> <コピー先>でコピー
  - `rm`でファイルを削除
  - `mv`でファイル名変更
  - `mkdir`でフォルダを作成
  - `rmdir`で中身が空のフォルダを削除
  - `cat`でファイルの中身を表示

- `vim`とその他コマンド
  - `vi`または`vim`でテキストエディタを開ける
  - `i`で，`vim`で編集できる.  閉じるときは`escape`
  - `vim`を閉じるときは`:wq`または`:aqw`
  - `ssh`でサーバーに接続し，`logout`で接続解除
  - `grep`で指定した単語の出現回数を数える
  - `Ctrl`+`C`で強制終了

学習を行うことで`saved` フォルダ内に作成される`.pth`を使って，推薦を行う．`.pth`は`PyTorch`のモデルデータである．\
`Hyperparameter` の設定は，以下を確認．
### `RecBole`における`Hyperparameter`設定
`RecBole`の[公式ドキュメント](https://recbole.io/docs/user_guide/usage/parameter_tuning.html)より詳細を確認できる．
- `Training Settings(訓練設定)`
  - `learning_rate`：学習率
  - `train_batch_size`：一度に計算するデータのサイズ
  - `learner`：最適化アルゴリズム
  - `epochs`：学習回数
- `Model Settings(モデル設定)`
  - `embedding_size`：ベクトルの次元数
  - `n_layers`：ニューラルネットワークの層の数
  - `droptout_prob`：過学習防止のため，ランダムに計算を休止する割合
- `Evaluation Settings(評価設定)`
  - `topk`：推薦リストに出す件数
  - `metrics`：評価指数の種類．例として，`Recall`，`NDCG`
  - `valid_metrics`：どの指標を基準としてモデルを選ぶか．例として，`Recall@10`
- `Dataset Settings(データセット設定)`
  - `load_col`：データとして扱う列
  - `USER_ID_FIELD`：ユーザーIDの列名

### その他まとめ
- `Parameters`と`Hyperparameters`
  - `Parameters`とは，モデルが学習を通して更新する値．例として，ニューラルネットワークの重みやバイアス
  - `Hyperparameters`とは，学習を始める前に決めておく値．例として，`learning_reta(学習率)`，`epochs(学習回数)`
- 推論・評価指数\
評価指数は，[公式ドキュメント](https://recbole.io/docs/recbole/recbole.evaluator.metrics.html)より詳細を確認できる
  - `Recall`：再現率
  - `Precision`：適合率
  - `NDCG`：順位指数
  - `MRR`：正解がどれだけ高順位か
  - `Hit Ratio`：正解の割合
  - `AUC(Area Under Curve)`：`1.0`に近い程良い．ランダムで`0.5`となる．`ROC曲線`の下部の面積
  - `LogLoss(logistic loss)`：`0`に近い程良い．
  - 評価指数には，`Pointwise指標`と`Ranking指標`の2種類ある．前者では，点推定を行うため，1点ずつの正解不正解を計算．`AUC`や`LogLoss`などである．後者では，順位を評価する．`Recall`，`Precision`，`NDCG`などである．
- モデル\
モデル設定は，[公式ドキュメント](https://www.recbole.io/model_list.html)より詳細を確認できる．以下，例を紹介する
  - [`FM(Factorization Machines)`](https://www.recbole.io/docs/user_guide/model/context/fm.html)：複数のデータの特徴量を計算
  - `GRU4Rec`：ユーザーの行動履歴をもとに評価
  - `DeepFM`: `FM`にニューラルネットワークを組み入れた深層学習モデル

### stat.inkについて
[stat.ink](https://stat.ink/)とは，`splatoon3`におけるバトルの戦績データを収集，保存，解析，共有ができる非公式のサービスである．アカウントを作成することで，自らの戦績データをアップロードでき，それを含めて統計情報を分析できる．しかし非公式であるので，任天堂の`API`が更新されるたび，戦績をアップロードするために必要な`s3s.py`を書き換える必要があり，ユーザー数はそのたびに減少傾向にある．しかし依然として大規模なデータがリアルタイムで追加されており，解析するには十分なデータ数である．データ自体は，`stat.ink`に登録していなくても利用できる．また，`splatoon3`は更新頻度が高いゲームであるがゆえに，統計情報は直近のアップデート後のデータを参照する必要がある．古いデータだと，現在の環境と異なる可能性があるためである．\
データのダウンロードは以下からできる．
- [統計情報ダウンロード](https://stat.ink/downloads)
- [スプラトゥーン3のバトルのリザルト情報](https://dl-stats.stats.ink/splatoon-3/battle-results-csv/)
- [スプラトゥーン3のスキーマ](https://github.com/fetus-hina/stat.ink/wiki/Spl3-%EF%BC%8D-CSV-Schema-%EF%BC%8D-Battle)
- [スプラトゥーン3のステージ一覧](https://stat.ink/api-info/stage3)
- [スプラトゥーン3の武器一覧](https://stat.ink/api-info/weapon3)
- [スプラトゥーン3のギアパワー一覧](https://stat.ink/api-info/ability3)

`.yaml`にはパラメータの詳細を記す．`.yaml`とは，`.html`や`.xml`などのデータ形式のうちの一つ．\
`TensorBoard` で機械学習の結果を可視化，学習曲線をプロットできる．[公式ドキュメント](https://recbole.io/docs/user_guide/usage/use_tensorboard.html)に使い方が詳細に書かれている．使用する際は，以下のコマンドを実行
```
tensorboard --logdir ./logs
```
サーバーを使う際は，まず以下のコマンドを実行し，鍵を取得する．`.pub`が作成される．
```
ssh-keygen -t ED25519
```
これで，鍵（秘密鍵，公開鍵）を取得できる．`ssh xxx`でサーバーに接続する．


## 2025/12/22
`Python`は仮想環境(`venv`など)で行うことを推奨．以下のコマンドを実行し，仮想環境を構築する．
```
python -m venv rec_env
```
仮想環境を構築したら，以下のコマンドで仮想環境に入る．
```
source rec_env/bin/activate
```
基本的に`Python`はそれぞれに仮想環境を構築すると便利である．特に今回は，`RecBole`が動く`Python`のバージョンが古く(後述)，既存のライブラリ等を書き換える必要があるため，仮想環境の使用は必須となっている．

`recbole`のインストールは，以下のコマンドを実行する．
```
pip install recbole
```
`Python 3.13`だと，`Recbole`が対応していためエラーが発生．その為，仮想環境を使ったうえで，`Python3`のライブラリを直接書き換えることを試みる．\
また，`PyTorch`で学習の際，`Early Stopping`となることがある．これは，過学習を防ぐための早期終了である．\
`VS Code`は，拡張機能の`Remote - SSH`で接続することができる．エクスプローラーより，任意のフォルダを開いて使う．


## stat.inkのデータセットを利用した簡単なギア推薦
まずは，`RecBole`による動作を確かめるためにも，単純なギア推薦を行った．以下が開発の記録である．\
[RecBoleによるギア推薦　前半](/src/readme/simple_rec.md)


以上により，`stat.ink`のデータセットを用いて`RecBole`を動かすことに成功した．次に，本格的なギア推薦システムの作成に取り掛かった．


## 2026/01/18
自宅の`Mac mini M4`をバックエンドサーバーとして，Webアプリケーションの作成を試みた．`Fast API`を使用して，モダンで軽いWebアプリケーションにする．以下が，フォルダ構成である．
```
recboletest/
├── data/                   # 学習・推論に使用するデータセット (.interファイルなど)
├── saved/                  # 学習したモデル (.pth) の格納先
├── backend/                # バックエンド関連
│   ├── main.py             # FastAPIのメインプログラム
│   ├── requirements.txt    # Mac Mini側で必要なライブラリ一覧
│   ├── build_master.py     # データベースを作成するスクリプト
│   ├── splatoon3_master.db # ここに生成される (SQLite DB)
│   └── tests/              # APIテスト用スクリプト
├── notebooks/              # 実験用コード (Colabで使用していたものなど)
├── .gitignore              # GitHubに上げないファイルの設定
└── README.md
```
`main.py`は`Fast API`のメインプログラムであり，推論したデータを`JSON`形式で返す．\
`requirements.txt`は，使用するライブラリをまとめてインストールするときに有用である．\
ギアパワー，ブキ，ステージなどをまとめてデータベースとして管理するために，`SQLite`を使って`JSON`のマッピングテーブルを作成する．`build_master.py`が作成するスクリプトである．\
`.gitignore`には，リポジトリを整えるために制約を書き込む．

そうして，試みた結果であるが，なかなかうまくいかなかった．考えられる原因として，次のものがある．

原因
- デバイスの問題: 作成したモデル(`.pth`)は`CUDA`で作成したものだった．しかし，`Mac mini`では`MPS`で動かす必要があり，そこでミスマッチが発生．
- `RecBole`の問題: `RecBole`は最新の`PyTorch`に対応していなかったりと，そもそも扱いづらく，今回もそこでエラーが頻発した．
- `FM`モデルの原因: モデルの特性として，特徴量が埋まっていない(`NaN`の状態)だと，エラーが起きる．

以上の点で解決する必要がある．
- `GPU`の違いについては，`torch.load()`で`.pth`を読み込み，`Apple GPU`の`MPS`を指定する必要がある．\
- `RecBole`側の問題については，仮想環境の`Docker`を使用し，`Linux`と同じ環境にすることで，条件がクリアできる．\
- モデルについては，欠損が無いように`.inter`を作り直す必要がある．


## 2026/01/19
データセットと推薦アルゴリズムを変更した．\
これまでは，`predict = Pr(weapon, ability, mode, stage, weight, flag)`というものだった．下図を参照．
```
# これまで使用していたデータセットの形式
weapon_id:token
ability_id:token
mode:token
stage:token
weight:float
label:float
```
しかし，このままでは一つしかギアの推薦ができない．求めたいのは，ギアの組み合わせである．その為，データセットを改良し，以下のようにした．
```
# 今回作成したデータセットの形式
# m1, m2, m3 is main gear power
# s1, s2, s3, s4, s5, s6, s7, s8 is sub gear power
weapon_id:token
mode:token
stage:token
m1:token	m2:token	m3:token
s1:token	s2:token	s3:token　s4:token	s5:token	s6:token	s7:token	s8:token	s9:token	label:float
```
こうすることで，ギア同士の関係を考慮に入れた，ギアの組み合わせを学習し，推薦することができるようになる．しかし，従来の`FM`では二次の特徴量しか計算できないため，今回のように多次元のデータでは難しい．\
そこで，`FM`にニューラルネットワークを組み入れた，`DeepFM`をモデルとして使い，学習することにした．以下が学習の結果である．
```
best valid : OrderedDict({'auc': 0.4957, 'logloss': 0.7008, 'rmse': 0.5037})
test result: OrderedDict({'auc': 0.5104, 'logloss': 0.6974, 'rmse': 0.5021})
```
`auc`は0.5付近となり，ほとんどランダムに近しいものになった．なぜなのか．それは，`splatoon3`において，勝敗に関係するのは，ギアの組み合わせによるものよりも，個人の技量が圧倒的に大きいからである．

そこで，勝利時データ(`label=1.0`)のみを扱うこととした．しかし，勝利データ(`label=1.0`)のみでは，`DeepFM`で推論するときに`AUC`を求めることができない．\
そこで，以前と同じように負例サンプリングをすることで，データセット`.inter`を作り直した．以下が，学習の結果である．
```
best valid : OrderedDict({'auc': 0.9973, 'logloss': 0.0659, 'rmse': 0.1365})
test result: OrderedDict({'auc': 0.9974, 'logloss': 0.0625, 'rmse': 0.1327})
```
すると，`auc=0.9974`と，驚異的な制度を得られることができた．しかしこれは，単に勝利時のギアの組み合わせと，完全にランダムで作ったギアの組み合わせを分類しているに過ぎない．一応このとき作成できた`DeepFM-xxx.pth`で推論した結果を載せる．
```
--- Top 10 Gear Sets for 52gal ---
Rank 1 (Score: 1.0000)
  Main: special_power_up, special_power_up, special_power_up
  Sub : quick_super_jump:0.6, special_power_up:0.9, sub_resistance_up:0.3
------------------------------
Rank 2 (Score: 1.0000)
  Main: swim_speed_up, intensify_action, drop_roller
  Sub : ink_saver_main:0.6, sub_resistance_up:0.9, swim_speed_up:0.3
------------------------------
Rank 3 (Score: 1.0000)
  Main: intensify_action, intensify_action, swim_speed_up
  Sub : ink_resistance_up:0.3, ink_saver_main:0.3, special_saver:0.6, swim_speed_up:0.3
------------------------------
Rank 4 (Score: 1.0000)
  Main: swim_speed_up, swim_speed_up, swim_speed_up
  Sub : quick_super_jump:0.3, special_power_up:0.6, sub_resistance_up:0.3, swim_speed_up:0.6
------------------------------
Rank 5 (Score: 1.0000)
  Main: swim_speed_up, quick_respawn, comeback
  Sub : ink_resistance_up:0.3, quick_respawn:0.3, quick_super_jump:0.3, swim_speed_up:0.9
------------------------------
```
すべて`score=1.0000`となり，推薦としては失敗している．しかし，推薦されたギアを見てみると，内容としては今までで一番良い推薦となっている．ただ，これは単に使われているギアセットを並べたものに過ぎない．

そこで，負例サンプリングのみではなく，実負例データを混ぜることで，良い感じになるのではないかと考えた．\
まずは，1:9の割合で，実負例データを少し混ぜた，その時の学習の結果と，推論の結果を以下に報告する．
```
best valid : OrderedDict({'auc': 0.9485, 'logloss': 0.2212, 'rmse': 0.2493})
test result: OrderedDict({'auc': 0.9439, 'logloss': 0.2322, 'rmse': 0.2553})
```
```
--- Top 10 Gear Sets for 52gal ---
Rank 1 (Score: 0.9763)
  Main: special_power_up, ninja_squid, tenacity
  Sub : special_charge_up:0.9, special_power_up:0.9, swim_speed_up:0.9
------------------------------
Rank 2 (Score: 0.9706)
  Main: sub_power_up, sub_power_up, sub_power_up
  Sub : ink_recovery_up:0.3, sub_power_up:0.6, swim_speed_up:0.9
------------------------------
Rank 3 (Score: 0.9674)
  Main: sub_power_up, sub_power_up, ink_saver_sub
  Sub : ink_recovery_up:0.6, ink_saver_sub:0.9, sub_power_up:0.3
------------------------------
Rank 4 (Score: 0.9656)
  Main: sub_power_up, sub_power_up, sub_power_up
  Sub : sub_power_up:0.9, swim_speed_up:0.9
------------------------------
Rank 5 (Score: 0.9651)
  Main: sub_power_up, sub_power_up, sub_power_up
  Sub : ink_recovery_up:0.6, ink_saver_sub:0.9, sub_power_up:0.3
------------------------------
```
また，実負例データと負例サンプリングを3:7の割合にすると，以下のようになった．
```
best valid : OrderedDict({'auc': 0.8501, 'logloss': 0.3981, 'rmse': 0.3618})
test result: OrderedDict({'auc': 0.8434, 'logloss': 0.4049, 'rmse': 0.364})
```
```
--- Top 10 Gear Sets for 52gal ---
Rank 1 (Score: 0.9236)
  Main: comeback, ninja_squid, stealth_jump
  Sub : ink_resistance_up:0.3, special_saver:0.9, sub_resistance_up:0.6, swim_speed_up:0.9
------------------------------
Rank 2 (Score: 0.9124)
  Main: comeback, ninja_squid, stealth_jump
  Sub : quick_super_jump:0.9, special_saver:0.3, sub_resistance_up:0.6, swim_speed_up:0.9
------------------------------
Rank 3 (Score: 0.9109)
  Main: quick_respawn, opening_gambit, stealth_jump
  Sub : ink_resistance_up:0.6, quick_respawn:0.9
------------------------------
Rank 4 (Score: 0.9097)
  Main: drop_roller, respawn_punisher, tenacity
  Sub : intensify_action:0.3, quick_respawn:0.3, quick_super_jump:0.9, sub_resistance_up:0.3
------------------------------
Rank 5 (Score: 0.9096)
  Main: ninja_squid, opening_gambit, stealth_jump
  Sub : quick_super_jump:0.9, special_saver:0.3, sub_resistance_up:0.6, swim_speed_up:0.9
------------------------------
```
どれも納得のいく答えになっているが，もう一押しする．今はブキのみを指定しているので，ブキ，ルール，ステージを指定して推薦してみる．
`inference.py`を更新して，推論した結果が以下である．
