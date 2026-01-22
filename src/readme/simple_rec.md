## 2025/12/23
`Nintendo Switch`のゲームソフト`Splatoon3`において，ブキ，ルール，ステージを選択したら，おすすめのギアパワーを教えてくれる推薦システムを作成する．

まずは，`dataset`を作成する．`Splatoon3`の詳細なバトル結果を入手したいので，`stat.ink`よりデータをダウンロードする．[stat.ink統計情報ダウンロード](https://stat.ink/downloads)からダウンロードできる．\
`Splatoon3`では，各シーズンごとに調整が入るため，直近1週間ほどのデータからデータセットを作成する．試しに直近1週間分(`2025-12-15`から`2025-12-21`)のデータでデータセットを作成したら，約66万件と十分なデータを得ることができた．

データセットは，`weapon_id(ブキ)`，`mode(ルール)`，`stage(ステージ)`，`ability_id(ギアパワー)`，`label(勝ち負け)`をそれぞれ記述する．勝ち負けについては，勝ちを`label=1.0`，負けを`label=0.0`として`flag`を作成する．\
`trainer.py`で作成したデータセット`splatoon3.inter`を用いて，モデルを作成する．`train.py`と`config.yaml`を使って，サーバー上で実行してモデルを作成する．

以下のようにディレクトリを構成した．
```
myproject/
├── dataset/
│   └── splatoon3/
│   │   └── splatoon3.inter  # 作成したデータセット
├── config.yaml              # 設定ファイル
├── train.py                 # 学習用スクリプト
├── predict.py               # 推論用スクリプト
├── saved/
└── └── xxx.pth              # PyTorchのモデルはここに作成される
```


## 2026/01/06
### ver 1.0
`run.py`を実行してできた`.pth`ファイルをもとに，推論用の`predict.py`を使用して，精度を確かめた．当初の予測精度と推論の結果は以下のようになった．
```
best valid : OrderedDict({'auc': 0.608, 'logloss': 0.6709})
test result: OrderedDict({'auc': 0.6055, 'logloss': 0.6727})
```
```
sshooter
1: comeback             (Score: 0.5176)  ##
2: ink_resistance_up    (Score: 0.4962)  #
3: stealth_jump         (Score: 0.4809)  #####
4: drop_roller          (Score: 0.4789)  #
5: special_saver        (Score: 0.4789)  ###
6: special_charge_up    (Score: 0.4752)  #####
7: swim_speed_up        (Score: 0.4675)  #####
8: quick_super_jump     (Score: 0.4639)  ###
9: object_shredder      (Score: 0.4609)  #
10: quick_respawn        (Score: 0.4563) ####

liter4k
1: drop_roller          (Score: 0.5592)  #
2: special_saver        (Score: 0.5394)  ##
3: haunt                (Score: 0.5373)  #
4: comeback             (Score: 0.5365)  #
5: respawn_punisher     (Score: 0.5296)  ####
6: object_shredder      (Score: 0.5205)  ###
7: ink_resistance_up    (Score: 0.5119)  #
8: ninja_squid          (Score: 0.5087)  #
9: stealth_jump         (Score: 0.5077)  #
10: ink_saver_main       (Score: 0.5068) ##
```
私のゲームの経験から，予測が正しいと思うものには`#`の数でチェックを入れている．また，この時使用していた`config.yaml`の設定は，`model: FM, learning_rate: 0.001, train_batch_size: 2048, epochs: 50`であった．

### UI作成
次に，UI作成を試みた．[`Google Colab`](https://colab.research.google.com/)でノートブックを新たに作成した．`recbole.jpynb`とした．まず，以下をセルで実行して，フォルダを作成する．
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
myproject/
├── dataset/
│   └── splatoon3/
│       └── splatoon3.inter  # 作成したファイル
└── xxx.pth                  # PyTorchのモデル
```
`localtunnel`を使うには，以下のコマンドをセルで実行する．まず，`グローバルIPアドレス`を取得する．
```
!curl ipv4.icanhazip.com
```
表示されたIPアドレスは，コピーする．\
次に，`localtunnel`をバックグラウンドで起動する．
```
streamlit run app.py &
```
最後に，`localtunnel`を起動する．ポート8501を指定する．
```
npx localtunnel --port 8501
```
表示された`https://xxx.loca.lt`にアクセスし、IPアドレスを入力する．\
以下に作成したUIのスクリーンショットを以下に掲載する．これは何度か試したのちに作成したものであるので，上の結果とは一致しない．
<p align="center">
  <img src="images/sample.png" width="600" alt="sample.png">
</p>

非常にシンプルな見た目となっているが，`Google Colab`を使って，ひとまずUIとして形にできた．

`Google Drive`にマウントするには，以下をセルで実行．
```
from google.colab import drive
drive.mount('/content/drive')
```

### ver 2.0
ver 1.0 で作成したデータセットの`splatoon3.inter`には致命的なデータの欠落があった．それは，ユーザーのギアパワーが考慮されていない点である．ユーザーのギアパワーが考慮されていないため，ギアパワーがすべて同じ値として処理されていた．\
スプラトゥーン3では，メインギアパワー(以下メインGP)の`1.0*3 = 3.0`とサブギアパワー(以下サブGP)の`0.3*3*3 = 2.7`の合計'5.7GP'が用意されている．例えば，以下のような組み合わせがある．
```
# 例えば，52gal(52ガロン)のギア構成
swim_speed_up = 1.6         # イカダッシュ速度アップ
quick_respawn = 1.3         # 復活時間短縮
stealth_jump = 1.0          # ステルスジャンプ
ink_saver_main = 0.9        # インク効率アップ(メイン)
quick_super_jump = 0.3      # スーパージャンプ時間短縮
intensify_action = 0.3      # アクション強化
special_power_up = 0.3      # スペシャル性能アップ

main GP = 1.0 + 1.0 + 1.0 = 3.0
sub GP = 0.6 + 0.3 + 0.9 + 0.3 + 0.3 + 0.3 = 2.7
total GP = 5.7
```
そのため，それらを考慮する必要があった．新たに，項目`weight`を追加し，各ブキに重みをつけてデータセットを作成した．上のような各ギアパワーの詳細は，[API情報:ギアパワー](https://stat.ink/api-info/ability3)から詳しく見ることができる．\
`trainer.py`を更新して，新たなデータセットを作成した．`config.yaml`も更新した．`model: FM, learning_rate: 0.001, train_batch_size: 2048, epochs: 50`で行った．結果を以下に報告する．
```
best valid : OrderedDict({'auc': 0.6081, 'logloss': 0.671})
test result: OrderedDict({'auc': 0.6047, 'logloss': 0.6731})
```
`AUC`が`0.6081`となり，少しだけ改善したが，ほとんど変わらなかった．今回の学習により，`FM-Jan-05-2026_02-15-20.pth`が生成された．`predict.py`も更新し，推論を行った結果を以下に報告する．
```
sshooter
1: special_saver             (Score: 0.7831)  ###
2: ink_saver_sub             (Score: 0.7680)  ##
3: run_speed_up              (Score: 0.7541)  ##
4: sub_power_up              (Score: 0.7523)  ##
5: ink_resistance_up         (Score: 0.7490)  ##
6: stealth_jump              (Score: 0.7456)  #####
7: special_charge_up         (Score: 0.7439)  #####
8: ninja_squid               (Score: 0.7254)  ##
9: ink_saver_main            (Score: 0.7228)  ####
10: respawn_punisher          (Score: 0.7166) #

liter4k
1: special_saver             (Score: 0.7650)  ##
2: sub_power_up              (Score: 0.7434)  #####
3: respawn_punisher          (Score: 0.7231)  ####
4: run_speed_up              (Score: 0.7195)  ###
5: ink_saver_sub             (Score: 0.7141)  ###
6: ink_recovery_up           (Score: 0.7052)  ###
7: thermal_ink               (Score: 0.7037)  ##
8: drop_roller               (Score: 0.7032)  #
9: ninja_squid               (Score: 0.7017)  #
10: opening_gambit            (Score: 0.7008) #
```
前回同様に，私の評価値と合致しているかどうかを，`#`の数で表している．"liter4k"はかなり改善したように思えるが，"sshooter"の方はいまいちと言った感じだった．以上より，結果だけ見ると，重みを追加しただけではあまり精度に変化は無いようだった．

### ver 3.0
今度は，精度向上のために，負例サンプリングの導入を考えた．装備していたギアだけではなく，装備していなかったギアを負け扱い(具体的には`flag = 0.0`とする)にした．負例の対象は，装備していなかったギアからランダムに3つほど選んで決めた．`trainer.py`を更新し，`.inter`ファイルを作り直した．\
学習した結果は以下のようになった．
```
best valid : OrderedDict({'auc': 0.766, 'logloss': 0.5151})
test result: OrderedDict({'auc': 0.7656, 'logloss': 0.5159})
```
`AUC`が`0.766`となり，ver1.0，ver2.0よりも改善したのがわかる．また，今回の学習で`FM-Jan-05-2026_03-17-27.pth`が生成された．\
次に，`predict.py`を改変した．使用したギアの特化度を偏差として計算し，それを考慮に入れた．すなわち，よく使われるギアは負の値に，珍しいギアは正の値に補正された．以下に推論結果を報告する．
```
52gal
順位   | ギアパワー名                    | 予測スコア      | 特化度(偏差)
---------------------------------------------------------------------------
   1 | run_speed_up              | 0.1917 | +0.0283  ##
   2 | intensify_action          | 0.2138 | +0.0276  #####
   3 | sub_resistance_up         | 0.1428 | +0.0139  ##
   4 | ink_resistance_up         | 0.2224 | +0.0071  ##
   5 | opening_gambit            | 0.0717 | +0.0061  ###
   6 | thermal_ink               | 0.0706 | +0.0020  #
   7 | quick_respawn             | 0.2196 | +0.0019  ####
   8 | quick_super_jump          | 0.3286 | +0.0016  ###
   9 | ninja_squid               | 0.0452 | -0.0032  ##
  10 | stealth_jump              | 0.0686 | -0.0062  #####
  11 | drop_roller               | 0.0646 | -0.0092  #
  12 | special_saver             | 0.1950 | -0.0097  ##
  13 | comeback                  | 0.0622 | -0.0107  ##
  14 | respawn_punisher          | 0.0626 | -0.0118  #
  15 | ink_recovery_up           | 0.1506 | -0.0124  ##
```
また，私による個人的な評価を以前と同様`#`の数で表示した．結果を見るに，実践で使えそうな精度に仕上がってきたと感じる．特に`intensify_action(アクション強化)`が推薦の上位に入ったのは良かった．一方，`thermal_ink(サーマルインク)`などの，短射程シューター(52ガロンなど)ではほとんど使われないようなギアが入っているのが少し気がかりであった．よって，まだ改善の余地があるように思える．\

## 2026/01/08
### ver 4.0
ここで，データセットを新たに作り直すことを考える．これまでのデータセットには，すべてのルール(`model: nawabari, area, yagura, hoko, asari`)，すべてのランク帯(`lobby: regular, bankara_challenge, bankara_open, xmatch, splatfest_challenge, splatfest_open, event`)が含まれていた．\
しかし，これらのユーザーが皆100%本気でゲームをプレイしているわけではなく，そこには明らかな偏りがある．そこで，比較的緩くプレイすることが多い`model: nawabari`を除き，学習に使用するデータは，全ユーザのうち上位10%ほどの`lobby: xmatch`のみを採用することにする．\
以上より，`trainer.py`を改変し，データセットを作り直した．作成したデータセットのサイズは`203084`であった．以下に学習の結果を報告する．
```
best valid : OrderedDict({'auc': 0.8383, 'logloss': 0.4319})
test result: OrderedDict({'auc': 0.841, 'logloss': 0.4255})
```
`AUC`が`0.841`となり，かなり良いデータで学習できたことがわかる．`.yaml`はver2.0と同じである．また，今回の学習で`FM-Jan-08-2026_03-17-38.pth`が生成された．\
以下に`predict.py`を実行して得られた推論の結果を報告する．
```
52gal の特化度(リフト値)
順位  | ギアパワー名              | 予測スコア | 特化度(偏差)
---------------------------------------------------------------------------
   1 | intensify_action          | 0.6135 | -0.0882  #####
   2 | ink_resistance_up         | 0.5450 | -0.0924  ###
   3 | sub_resistance_up         | 0.5901 | -0.0982  ###
   4 | ink_recovery_up           | 0.5232 | -0.1089  ####
   5 | ink_saver_sub             | 0.5148 | -0.1202  ###
   6 | opening_gambit            | 0.5345 | -0.1237  ###
   7 | quick_super_jump          | 0.4810 | -0.1437  ###
   8 | quick_respawn             | 0.4450 | -0.1590  ###
   9 | stealth_jump              | 0.5429 | -0.1681  #####
  10 | ink_saver_main            | 0.4159 | -0.1847  ####
  11 | swim_speed_up             | 0.4254 | -0.1916  #####
  12 | run_speed_up              | 0.4488 | -0.1949  ##
  13 | special_saver             | 0.5390 | -0.2072  ##
  14 | special_charge_up         | 0.4193 | -0.2224  ##
  15 | comeback                  | 0.4837 | -0.2281  ##
```
今までで一番良い結果になった．ブレの大きいブキである`52gal(52ガロン)`で使われがちな`intensify_action(アクション強化)`は一位になることができた．`ink_recovry_up(インク回復力アップ)`も上位に来た．しかし，`stealth_jump(ステルスジャンプ)`，`swim_speed_up(イカダッシュ速度アップ)`は少し順位が低くなった．これらの結果には複合的な要因があると思われる．例えば，特化度を偏差で補正しているために，過度にバイアスが入ってしまっていることが挙げられる．また，ルールによっても大きく変わるだろうと考えられる．データセットも，負例サンプリングがランダムに3種選んでいるだけなので，10種などに増やしてみても良いかもしれない．

### ver 4.1
データセットを少し改良した．まず，サンプル数を2倍にした．`2025-12-15`から`2025-12-28`までの2週間分を元データとした．また，負例サンプリングをブキ当たり5個から12個に増やした．そうすることでよりランダムになりバイアスが消えると思われる．以下に学習の結果を報告する．
```
best valid : OrderedDict({'auc': 0.898, 'logloss': 0.2979})
test result: OrderedDict({'auc': 0.8992, 'logloss': 0.2961})
```
`AUC: 0.899`，`LogLoss: 0.29`となり，非常に高水準に推論ができるようになった．また，今回の学習で`saved/FM-Jan-08-2026_03-48-50.pth`が作成された．\
次に`predict.py`を少し変える．変えると言っても，特化度を用いた結果と，用いない素のままの結果の両方を表示するようにしただけである．`52gal(52ガロン)`についての結果は以下のようになった．
```
52gal 推論結果レポート
 [1. 総合おすすめ順] (汎用的に評価が高いもの)
順位   | ギアパワー名                    | 予測スコア
-----------------------------------------------------------------
   1 | intensify_action          | 0.7349  #####
   2 | special_charge_up         | 0.7191  ##
   3 | sub_resistance_up         | 0.7145  ###
   4 | ink_resistance_up         | 0.7066  ###
   5 | sub_power_up              | 0.7003  ##
   6 | special_saver             | 0.6880  ##
   7 | drop_roller               | 0.6759  #
   8 | ink_saver_sub             | 0.6713  ###
   9 | thermal_ink               | 0.6686  #
  10 | quick_respawn             | 0.6640  ###

 [2. 特化度順]
順位   | ギアパワー名                    | 特化度(偏差)
-----------------------------------------------------------------
   1 | quick_respawn             | +0.0898  ###
   2 | intensify_action          | +0.0892  #####
   3 | ink_saver_sub             | +0.0106  ###
   4 | run_speed_up              | +0.0060  ##
   5 | sub_power_up              | +0.0057  ##
   6 | sub_resistance_up         | +0.0016  ###
   7 | ink_resistance_up         | -0.0150  ###
   8 | ink_recovery_up           | -0.0194  ####
   9 | special_charge_up         | -0.0312  ##
  10 | special_saver             | -0.0372  ##
```
以上のような結果になった．良いとも悪いとも言えない結果になった．`stealth_jump`と`swim_speed_up`が消えてしまったのが気がかりである．

## 2026/01/09
### ver 4.2
`predict.py`を変更し，`ブキ(weapon)`に加えて`ルール(mode)`と`ステージ(stage)`を選択してギアの推薦を受けるようにした．以下に，`52gal`の`area`で`yunohana`の推薦結果を報告する．
```
52gal ギア予測ランキング (ルール:area / ステージ:yunohana)
予測順位 | ギアパワー名                    | 予測スコア      | 特化度(偏差)      | (特化順位)
--------------------------------------------------------------------------------------------
     1位 | quick_super_jump          | 0.5440     |     -0.0503 |  17位
     2位 | swim_speed_up             | 0.2456     |     -0.1319 |  26位
     3位 | ink_resistance_up         | 0.2002     |     -0.0729 |  20位
     4位 | special_charge_up         | 0.1701     |     -0.1137 |  23位
     5位 | ink_saver_sub             | 0.1475     |     -0.0518 |  18位
     6位 | ink_recovery_up           | 0.1153     |     -0.0793 |  21位
     7位 | quick_respawn             | 0.1096     |     -0.0168 |  13位
     8位 | special_saver             | 0.1073     |     -0.0954 |  22位
     9位 | ink_saver_main            | 0.0999     |     -0.1225 |  25位
    10位 | special_power_up          | 0.0985     |     -0.1199 |  24位
    11位 | intensify_action          | 0.0968     |     -0.0402 |  15位
    12位 | sub_resistance_up         | 0.0886     |     -0.0547 |  19位
    13位 | run_speed_up              | 0.0746     |     -0.0428 |  16位
    14位 | sub_power_up              | 0.0674     |     -0.0258 |  14位
    15位 | stealth_jump              | 0.0022     |     -0.0037 |  12位
--------------------------------------------------------------------------------------------
```
結果としては，良くない推薦となった．`quick_super_jump`はそこまで使われるギアではないし，`stealth_jump`はもっと使われてよいはずである．\
そこで，`.inter`を調べ，データセット不足なのかを検証した．`grep`コマンドを使い，`quick_super_jump`と`stealth_jump`の出現回数を調べた．
```
grep "52gal" dataset/splatoon3_xmatch/splatoon3_xmatch.inter | grep "area" | grep "stealth_jump" | grep "1.0" | wc -l
grep "52gal" dataset/splatoon3_xmatch/splatoon3_xmatch.inter | grep "area" | grep "quick_super_jump" | grep "1.0" | wc -l
```
結果は，`398`と`203`となった．そこで，先ほどの結果がおかしいことがわかる．考えられる原因としては，`quick_super_jump`はメインだけではなくサブとしてもギアが付くのに対し，`stealth_jump`は`1.0`か`0.0`の二択である．そのため，現状の負例サンプリングを用いているままだと，`stealth_jump`のようなギアでは，この`FM`モデルにおいて不利に働いてしまう．以下で，`grep`を使い，今度は負例サンプリングとしての出現回数を調べた．
```
grep "52gal" dataset/splatoon3_xmatch/splatoon3_xmatch.inter | grep "area" | grep "stealth_jump" | grep "0.0" | wc -l
grep "52gal" dataset/splatoon3_xmatch/splatoon3_xmatch.inter | grep "area" | grep "quick_super_jump" | grep "0.0" | wc -l
```
結果は`227`と`162`となった．よって，やはり負例サンプリングが原因であったことが確認できた．

### 負例サンプリングについて
負例サンプリングをしないでデータセットを作成したらどうなるか，試してみた．`splatoon3_xmatch_onlywin.inter`とし，学習したところ，結果は以下のようになった．
```
best valid : OrderedDict({'auc': 0.6474, 'logloss': 0.6503})
test result: OrderedDict({'auc': 0.6447, 'logloss': 0.6509})
```
先ほどは`auc: 0.898`であったことを考えると，負例サンプリングは必要な処理であることが分かった．

では，どうすればよいか．負例サンプリングはこのまま活用し，かつ`stealth_jump`などのギアに負のバイアスがかからないようにしたい．
