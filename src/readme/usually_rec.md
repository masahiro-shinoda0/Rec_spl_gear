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
- `GPU`の違いについては，`torch.load()`で`.pth`を読み込み，`Apple GPU`の`MPS`を指定する必要がある．
- `RecBole`側の問題については，仮想環境の`Docker`を使用し，`Linux`と同じ環境にすることで，条件がクリアできる．
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
