# recboletest
[`RecBole`](https://www.recbole.io/index.html)を使ったゼミの課題に取り組むうえで，気になったことを以下に書き記す．\
2025/12/08に作成．`sinki.py`と`sinki.yaml`は動かない．\
2025/12/22から，[`stat.ink`](https://stat.ink/)のデータを利用した推薦システムの作成を開始．


## 2025/12/08
`conda`を使用するために`Anaconda`を使用する．
`Anaconda Prompt` を起動し，以下のコマンドを実行．
```
conda activate recbole
```
`Linux`のコマンドをメモする．`cd` でファイルのある階層に移動できる．\
`python xxx.py` で`Python`を実行できる．`pwd`で現在の階層を表示する．\
`dir` コマンドで，作成した `saved` フォルダを見つける．\
`touch`でファイルを作成，`mkdir`でフォルダを作成できる．`cat`でファイルの中身を表示．`vi`または`vim`でテキストエディタを開ける．\
`saved` フォルダ内にある`.pth`を使って，新規ユーザの推薦を行う．`.pth`は`PyTorch`のモデルデータである．\
`hyper parameter` の設定は，レファレンスを読んで理解する．\
`epoch`はエポック数であり，これは学習する回数である．エポック数が大きいとき，自前のノートパソコンだと遅すぎて終わらないため，研究室のサーバーを利用する．\
`.yaml`にはパラメータの詳細を記す．`.yaml`とは，`.html`や`.xml`などのデータ形式のうちの一つ．\
学習は，何度か試すことで，一番良いパラメータを探す．\
`TensorBoard` で機械学習の結果を可視化，学習曲線をプロットできる．使用する際は，以下のコマンドを実行
```
tensorboard --logdir ./logs
```
自前のノートパソコンだと厳しいので，サーバーを使う．\
サーバーを使う際は，まず以下のコマンドを実行し，鍵を取得する．`.pub`が作成される．
```
ssh-keygen -t ED25519
```
これで，鍵（秘密鍵，公開鍵）を取得できる．`ssh xxx`でサーバーに接続する．


## 2025/12/22
`Python`は仮想環境(`venv`など)で行うことを推奨．以下のコマンドを実行し，仮想環境を構築する．
```
python -m venv rec_env
source rec_env/bin/activate
```
`recbole`のインストールは，以下のコマンドを実行する．
```
pip install recbole
```
`Python 3.13`だと，新しすぎて`PyTorch`などが対応していなく，エラーが起きる．その為，仮想環境を使ったうえで，`Python3`のライブラリを直接書き換える．\
`PyTorch`で学習していると，早期終了`Early Stopping`することがある．これは，過学習を防ぐためである．\
サーバーでは，`vim`を使ってプログラムファイルを作る．`touch test.py`と適当にファイルを作ったら，`vi test.py`としてターミナル上で`vim`を開く．`i`と打つことで，編集ができる．やめるときは，`escape`を押し，閉じるときは`:wq`とする．`:aqw`とする場合もある．サーバーを閉じるときは，`exit`とする．\
`VS Code`は，拡張機能の`Remote - SSH`で接続することができる．エクスプローラーより，任意のフォルダを開いて使う．


## 2025/12/23
`Nintendo Switch`のゲームソフト`Splatoon 3`において，ブキとルール，ステージを選択したら，おすすめのギアパワーを教えてくれる推薦システムを作りたい．\
まずは，`dataset`を作成する．`splatoon 3`の詳細なバトル結果を入手したいので，`stat.ink`よりデータをダウンロードする．[ここ](https://stat.ink/downloads)からダウンロードできる．\
`splatoon 3`では，各シーズンごとに調整が入るため，直近1週間ほどのデータからデータセットを作成する．ためしに直近1週間分のデータでデータセットを作成したら，約66万件と十分なデータを得ることができた．\
`trainer.py`で作成したデータセット`splatoon3.inter`を用いて，モデルを作成する．`train.py`と`config.yaml`を使って，サーバー上で実行してモデルを作成する．\
以下のようにディレクトリを構成した．
```
myproject/
├── dataset/
│   └── splatoon3/
│       └── splatoon3.inter  # 作成したファイル
├── config.yaml              # 設定ファイル
├── train.py                 # 学習用スクリプト
├── predict.py               # 推論用スクリプト
├── venv/
│   └── .../                 # 仮想環境
├── saved/
└── └── xxx.pth              # PyTorchのモデル
```

## 2026/01/04
`ModuleNotFoundError: No module named 'ray'`このエラーが起きたら，以下のコマンドを実行する．
```
pip install ray
```
`ModuleNotFoundError: No module named 'pyarrow'`このエラーが起きたら，以下のコマンドを実行する．
```
pip install pyarrow
```
`ModuleNotFoundError: No module named 'pydantic'`このエラーが起きたら，以下のコマンドを実行する．
```
pip install pydantic
```
仮想環境のアクティベートはこれを実行．
```
source bin/activate
```
うまくいかないときは，以下を試すと動くらしい．仮想環境で行う．まず，`torch`のバージョンを指定して実行．
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
次に，`PyTorch`の一部のコードを改変する．`venv`の場合，`env/lib/python3.13/site-packages/recbole/trainer/trainer.py`の583行目を以下のように変更．\
`checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)`

## 2026/01/06
`run.py`を実行してできた`.pth`ファイルをもとに，推論用の`predict.py`を使用して，精度を確かめた．当初の予測精度と推論の結果は以下のようになった．
```
auc : 0.6073    logloss : 0.6711

05 Jan 00:17    INFO  Finished training, best eval result in epoch 23
```
```
--- sshooter のおすすめギアパワー TOP10 ---
1: comeback             (Score: 0.5176)
2: ink_resistance_up    (Score: 0.4962)
3: stealth_jump         (Score: 0.4809)
4: drop_roller          (Score: 0.4789)
5: special_saver        (Score: 0.4789)
6: special_charge_up    (Score: 0.4752)
7: swim_speed_up        (Score: 0.4675)
8: quick_super_jump     (Score: 0.4639)
9: object_shredder      (Score: 0.4609)
10: quick_respawn        (Score: 0.4563)

--- liter4k のおすすめギアパワー TOP10 ---
1: drop_roller          (Score: 0.5592)
2: special_saver        (Score: 0.5394)
3: haunt                (Score: 0.5373)
4: comeback             (Score: 0.5365)
5: respawn_punisher     (Score: 0.5296)
6: object_shredder      (Score: 0.5205)
7: ink_resistance_up    (Score: 0.5119)
8: ninja_squid          (Score: 0.5087)
9: stealth_jump         (Score: 0.5077)
10: ink_saver_main       (Score: 0.5068)
```
次に，UI作成を試みた．`Google Colab`でノートブックを新たに作成した．`recbole.jpynb`とした．以下をセルで実行して，フォルダを作成する．
```
!mkdir -p dataset/splatoon3
```
また，ライブラリのインストールは，以下のコマンドをセルで実行する．
```
!pip install recbole streamlit pyngrok -q
!npm install -g localtunnel -q
```
フォルダ内に`app.py`を作成し，アプリケーションのメイン部分とする．`streamlit`というフレームワークを使い，ブラウザ上で動くアプリを作る．`localtunnel`を組み合わせて，一時的にWebアプリを外部公開する．\
`splatoon3.inter`と`.pth`ファイルを使い，アプリを動かす．`Colab`上のディレクトリ構造は以下のようである．
```
recbole_project/
├── dataset/
│   └── splatoon3/
│       └── splatoon3.inter  # 作成したファイル
└── xxx.pth                  # PyTorchのモデル
```
