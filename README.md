# recboleを用いた推薦システムの作成
[`RecBole`](https://www.recbole.io/index.html)を使ったゼミの課題に取り組むうえで，気になったことを以下に書き記す．

2025/12/08に作成．
2025/12/22から，[`stat.ink`](https://stat.ink/)のデータを利用した推薦システムの作成を開始．\
2026/01/06から，[`Google Colab`](https://colab.research.google.com/)上で動く[`Streamlit`](https://streamlit.io/)を利用したUIの作成を開始．\
2026/01/18から，[`FastAPI`](https://fastapi.tiangolo.com/)をバックエンドに使用したWebアプリケーションの作成を開始．


以下は`Git`の使い方をまとめたものである．\
[Gitの使い方まとめ](https://github.com/masahiro-shinoda0/AtCoder/blob/main/readme/HowToUseGitHub.md)


### RecBoleについて
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
 
### スプラトゥーン３とは
`2022`年に任天堂より発売された，アクションシューティングゲームである．`2015`年発売の初代スプラトゥーン，`2017`年発売のスプラトゥーン２に続く，三作目である．
ゲームの内容としては，三人称視点のシューティングゲーム(`TPS`)である．インクリングというプレイヤーを操作し，ブキからインクを放つことで，インクを塗って陣地を広げたり，相手を倒したりする．

本ゲームの特徴として，多くの`FPS`のように弾を打つかわりに，インクを打つというところがある．また，打ったインクで床や壁を塗り，そこをヒト状態からイカ状態に変化することで，潜ることができる．その特性上，多彩なブキが登場しており，ブキの数は`130`にもなる．
また，大きく分けて5つのルールがあることに加え，ステージも26種も存在する．その為，今回のギア推薦システムにおいて，入力をブキ，ルール，ステージとしている．

### ギアについて
以下の画像は，`splatoon3`におけるブキ，装備の選択画面である．下にブキ，アタマ，フク，クツとあるのがわかる．アタマ，フク，クツが装備である．

<p align="center">
  <img src="/src/readme/images/spl_gear_set.jpg" width="600" alt="spl_gear_set.jpg">
</p>

装備には`ギア`という効果が付く．以下に，上の画像の装備の部分だけを拡大したものを載せる．

<p align="center">
  <img src="/src/readme/images/spl_gear_up.jpg" width="600" alt="spl_gear_up.jpg">
</p>

この画像を例にして，ギアを説明する．**アタマに付いているギア**に注目する．左から`イカダッシュ速度アップ`，`スペシャル減少量ダウン`，`インク効率アップ`，`インク効率アップ`である．一番左の`イカダッシュ速度アップ`だけ少し大きいサイズなのがわかる．これがメインギアパワーであり，`GP(ギアパワー)`は`1.0`となる．その他の3つはサブギアパワーとなり，`GP`は`0.3`である．これがフク，クツにもそれぞれあるので，トータルギアパワーは`(1.0+0.3+0.3+0.3)*3=5.7`となる．

リンク
- [スプラトゥーン3公式サイト](https://www.nintendo.com/jp/switch/av5ja/index.html)
- [スプラトゥーン公式X(通称イカ研)](https://x.com/SplatoonJP)
- [スプラトゥーン3のギアを徹底解説 -イカクロ](https://www.ikaclo.jp/3/guides/gears)

推薦システムの作成手順として，以下のように進める．
1. [`stat.inkのAPI`](https://github.com/fetus-hina/stat.ink/wiki/Spl3-%EF%BC%8D-CSV-Schema-%EF%BC%8D-Battle)よりデータをダウンロード
2. データセットを作成する．`trainer.py`を用いて，`.inter`を作成．
3. 作成したデータセットと，`.yaml`，`.hyper`を用いて，学習をする．モデルを選定し，`run.py`より学習．
4. 学習結果を見て，そのモデルを使うか決める．不足なら，1. に戻る．
5. 作成したモデル`.pth`を用いて，推薦を行う．


### stat.inkについて
[stat.ink](https://stat.ink/)とは，`splatoon3`におけるバトルの戦績データを収集，保存，解析，共有ができる非公式のサービスである．アカウントを作成することで，自らの戦績データをアップロードでき，それを含めて統計情報を分析できる．しかし非公式であるので，任天堂の`API`が更新されるたび，戦績をアップロードするために必要な`s3s.py`を書き換える必要があり，ユーザー数はそのたびに減少傾向にある．しかし依然として大規模なデータがリアルタイムで追加されており，解析するには十分なデータ数である．データ自体は，`stat.ink`に登録していなくても利用できる．また，`splatoon3`は更新頻度が高いゲームであるがゆえに，統計情報は直近のアップデート後のデータを参照する必要がある．古いデータだと，現在の環境と異なる可能性があるためである．\
データのダウンロードは以下からできる．
- [統計情報ダウンロード](https://stat.ink/downloads)
- [スプラトゥーン3のバトルのリザルト情報](https://dl-stats.stats.ink/splatoon-3/battle-results-csv/)
- [スプラトゥーン3のスキーマ](https://github.com/fetus-hina/stat.ink/wiki/Spl3-%EF%BC%8D-CSV-Schema-%EF%BC%8D-Battle)
- [スプラトゥーン3のステージ一覧](https://stat.ink/api-info/stage3)
- [スプラトゥーン3の武器一覧](https://stat.ink/api-info/weapon3)
- [スプラトゥーン3のギアパワー一覧](https://stat.ink/api-info/ability3)

## 環境構築
最初に，`RecBole`を使用するために環境構築を行った．以下が記録である．\
[RecBoleの環境構築](/src/readme/environment.md)


## stat.inkのデータセットを利用した簡単なギア推薦
まずは，`RecBole`による動作を確かめるためにも，単純なギア推薦を行った．以下が開発の記録である．\
[RecBoleによるギア推薦　前半](/src/readme/simple_rec.md)


## DeepFMを用いたギアセットの推薦
以上により，`stat.ink`のデータセットを用いて`RecBole`を動かすことに成功した．次に，本格的なギア推薦システムの作成に取り掛かった．以下が開発の記録である\
[DeepFMによるギアセットの推薦](/src/readme/usually_rec.md)
