## 2025/12/08
`conda`を使用するために`Anaconda`を使用する．
`Anaconda Prompt` を起動し，以下のコマンドを実行．
```cmd
conda activate recbole
```
これは自前ノートパソコンで`RecBole`を使用するために用いたためであり，以降ではサーバーに仮想環境を構築して行ったため，`Anaconda`は使用していない．

`.yaml`にはパラメータの詳細を記す．`.yaml`とは，`.html`や`.xml`などのデータ形式のうちの一つ．\
`TensorBoard` で機械学習の結果を可視化，学習曲線をプロットできる．[公式ドキュメント](https://recbole.io/docs/user_guide/usage/use_tensorboard.html)に使い方が詳細に書かれている．使用する際は，以下のコマンドを実行
```cmd
tensorboard --logdir ./logs
```
サーバーを使う際は，まず以下のコマンドを実行し，鍵を取得する．`.pub`が作成される．
```cmd
ssh-keygen -t ED25519
```
これで，鍵（秘密鍵，公開鍵）を取得できる．`ssh xxx`でサーバーに接続する．


## 2025/12/22
`Python`は仮想環境(`venv`など)で行うことを推奨．以下のコマンドを実行し，仮想環境を構築する．
```cmd
python -m venv rec_env
```
仮想環境を構築したら，以下のコマンドで仮想環境に入る．
```cmd
source rec_env/bin/activate
```
基本的に`Python`はそれぞれに仮想環境を構築すると便利である．特に今回は，`RecBole`が動く`Python`のバージョンが古く(後述)，既存のライブラリ等を書き換える必要があるため，仮想環境の使用は必須となっている．

`recbole`のインストールは，以下のコマンドを実行する．
```cmd
pip install recbole
```
`Python 3.13`だと，`Recbole`が対応していためエラーが発生．その為，仮想環境を使ったうえで，`Python3`のライブラリを直接書き換えることを試みる．\
また，`PyTorch`で学習の際，`Early Stopping`となることがある．これは，過学習を防ぐための早期終了である．\
`VS Code`は，拡張機能の`Remote - SSH`で接続することができる．エクスプローラーより，任意のフォルダを開いて使う．


## 2026/01/04
初めて`RecBole`を用いるときは，以下のようなエラーが起きる．その時の対処を書き記す．
`ModuleNotFoundError: No module named 'ray'`このエラーが起きたら，以下のコマンドを実行する．
```cmd
pip install ray
```
`ModuleNotFoundError: No module named 'pyarrow'`このエラーが起きたら，以下のコマンドを実行する．
```cmd
pip install pyarrow
```
`ModuleNotFoundError: No module named 'pydantic'`このエラーが起きたら，以下のコマンドを実行する．
```cmd
pip install pydantic
```
仮想環境(`venv`など)の階層に入り，以下を実行することで仮想環境を使用できる．
```cmd
source bin/activate
```
うまくいかないときは，以下を試す．必ず仮想環境で行う．まず，`PyTorch`のバージョンを指定して実行．
```cmd
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
次に，`PyTorch`の一部のコードを改変する．`venv`の場合，`env/lib/python3.13/site-packages/recbole/trainer/trainer.py`の583行目を以下のように変更．\
`checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)`
