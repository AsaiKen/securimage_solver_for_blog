# Securimageのソルバ

AIのモデルについては[本家](https://github.com/clovaai/deep-text-recognition-benchmark)を参照して下さい。

事前準備
---

Ubuntu 18.04

```
$ sudo apt-get install build-essential libssl-dev libffi-dev python3-dev
$ python3 -mvenv venv
$ . venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt --no-cache-dir
```

Mac mojave

```
$ sudo xcrun cc
$ brew install libomp
$ python3 -mvenv venv
$ . venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

学習済モデルを利用する場合
---

Webサーバ起動
---

```
$ . venv/bin/activate
$ python3 web/app.py
```

http://localhost:5000/ にアクセスする。

学習済モデルを利用しない場合
---

Ubuntu 18.04で動作確認

サンプルデータの作成
---

```
$ python3 securimage/securimage.py
$ python3 securimage/resize.py
$ python3 securimage/create_gt_file.py
$ for model in 'train' 'test' 'validate'
do
  python3 create_lmdb_dataset.py --inputPath ../data/resize/ \
  --gtFile ../data/resize/$model/gt.txt \
  --outputPath ../data/resize/$model/
done
```

トレーニング
---

```
$ CUDA_VISIBLE_DEVICES=0 python3 train.py \
--train_data ../../data/resize/train --valid_data ../data/resize/validate \
--select_data / --batch_ratio 1 \
--Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC \
--sensitive
```

テスト
---

```
$ CUDA_VISIBLE_DEVICES=0 python3 test.py \
--eval_data ../data/resize/test \
--Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC \
--sensitive \
--saved_model saved_models/None-VGG-BiLSTM-CTC-Seed1111/best_accuracy.pth
```

Webサーバ起動
---

```
$ . venv/bin/activate
$ python3 web/app.py
```

http://localhost:5000/ にアクセスする。
