# References


## Dataset

* [ESC-50: Dataset for Environmental Sound Classification](https://github.com/karoldvl/ESC-50)




## 実装

* [ディープラーニングで音声分類](https://qiita.com/cvusk/items/61cdbce80785eaf28349#augmentation)
* [畳込みニューラルネットワークの基本技術を比較する　ー音でもやってみたー](https://qiita.com/shinmura0/items/858214154f889c05e4f4#%E3%83%A1%E3%83%AB%E3%82%B9%E3%83%9A%E3%82%AF%E3%83%88%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0%E3%81%AE%E5%8A%B9%E6%9E%9C)
* [さぁ、始まる、本気の異音検知器](https://qiita.com/shinmura0/items/58cbda22d6bbf1842edd)
* [「弱異常検知」と他の異常検知、違いはなにか？](https://qiita.com/shinmura0/items/1af83f5a5857d50cabc2)




## References

論文・実装・解説記事等へのリンク集


### 全般

* [Qiita 異常検知](https://qiita.com/tags/%E7%95%B0%E5%B8%B8%E6%A4%9C%E7%9F%A5/items)


#### 入門

* [Deep Learning Lab 異常検知入門](https://www.slideshare.net/shoheihido/deep-learning-lab-88299985)


#### 異常検知と変化検知

* [「異常検知と変化検知～機械学習プロフェッショナルシリーズ～」を読んで勉強会も開いて関連の情報とかも調べたのでまとめておく](https://www.st-hakky-blog.com/entry/2017/12/08/013205)



### DAD (Deep Anomaly Detection)

* [Deep Learningを用いた教師なし画像検査の論文調査 GAN/SVM/Autoencoderとか](https://www.slideshare.net/rist_inc/deep-learning-gansvmautoencoder-pdf)


#### AutoEncoder

* [Latent Space Autoregression for Novelty Detection](https://arxiv.org/abs/1807.01653)
* [Outlier detection using autoencoders](https://cds.cern.ch/record/2209085/files/Outlier%20detection%20using%20autoencoders.%20Olga%20Lyudchick%20(NMS).pdf)
* [Anomaly Detection with Robust Deep Autoencoders](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p665.pdf)
* [AutoEncoderでシンプルな音声の異常検知(Keras/TensorFlow)](https://qiita.com/shinkoizumi0033/items/f787845282d3d5d06d9a)
* [Keras AutoEncoder で異常検知をやってみる](http://cedro3.com/ai/keras-autoencoder-anomaly/)
* [Autoencoderによる異常検知](http://www.renom.jp/ja/notebooks/tutorial/clustering/anomaly-detection-using-autoencoder/notebook.html)
* [Autoencoderで異常検知](https://momentum.hatenablog.jp/entry/2018/11/16/001742)



#### VAE

* [Variational Autoencoder based Anomaly Detection using Reconstruction Probability](https://pdfs.semanticscholar.org/0611/46b1d7938d7a8dae70e3531a00fceb3c78e8.pdf?_ga=2.261807503.309185382.1559326519-165992685.1549479546)
* [Variational Auto Encoder 〜外れ値検知への応用〜](https://orizuru.io/blog/machine-learning/variational-auto-encoder/)


#### Unnormalized VAE

* Paper: [深層生成モデルによる非正則化異常度を用いた工業製品の異常検知](https://confit.atlas.jp/guide/event-img/jsai2018/2A1-03/public/pdf?type=in)
* [Variational Autoencoderを使った画像の異常検知　前編](https://qiita.com/shinmura0/items/811d01384e20bfd1e035)
* [Variational Autoencoderを使った画像の異常検知　後編　（塩尻MLもくもく会＃7）](https://qiita.com/shinmura0/items/6572d782ad21b15b004e)
* [Keras VAEの画像異常検出を理解する](http://cedro3.com/ai/keras-vae-anomaly/)


#### DOC

* Paper: [Learning Deep Features for One-Class Classification](https://arxiv.org/abs/1801.05365)
* Code: [PramuPerera/DeepOneClass](https://github.com/PramuPerera/DeepOneClass)
* [ディープラーニングを使った画像の異常検知　－論文と実装－](https://qiita.com/shinmura0/items/cfb51f66b2d172f2403b)
* [がんばる人のための画像検査機](https://qiita.com/shinmura0/items/7f4298b75d6b788bba80)
* [さぁ、始まる、本気の異音検知器](https://qiita.com/shinmura0/items/58cbda22d6bbf1842edd)


#### DAGMM

* Paper: [Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection](https://openreview.net/forum?id=BJJLHbb0-)
* Code: [danieltan07/dagmm](https://github.com/danieltan07/dagmm)
* [異常値検出手法 DAGMM の実装](https://qiita.com/ToshihiroNakae/items/5863fd70ed9afc5fd6cd)
    * [DAGMM Tensorflow implementation](https://github.com/tnakae/DAGMM)
* [Chainerで異常検知手法DAGMMを実装してみた](https://qiita.com/KaijiS/items/94f880ce066dab053c8f)
* [深層自己符号化器＋混合ガウスモデルによる教師なし異常検知](https://www.slideshare.net/ChihiroKusunoki/ss-98822807)
* [論文読みメモ： 深層自己符号化器＋混合ガウスモデルによる教師なし異常検知（その1）](http://cookie-box.hatenablog.com/entry/2018/05/18/004544)


#### GAN

* [異常検知とGANのまとめ(ベースとなっている論文のおさらい)](https://note.mu/koichirot11/n/n24151f3c7af8)
* [【論文読み】GANを利用した異常検知まとめ](http://habakan6.hatenablog.com/entry/2018/04/29/013200)
* [【異常検知】GANを用いた画像の異常検知](http://ni4muraano.hatenablog.com/entry/2018/08/14/174901)
* [最新研究紹介資料（異常検知）](https://www.slideshare.net/ShunsukeNAKATSUKA1/ss-106663876)


#### AnoGAN

* Paper: [Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://arxiv.org/abs/1703.05921)
* Code: [LeeDoYup/AnoGAN](https://github.com/LeeDoYup/AnoGAN)
* [AnoGANの論文を読んでMNISTの異常検知をしてみた](https://qiita.com/NakaokaRei/items/231ec4efe42dfe79d1ff)
* [GANによる異常検知手法 anoGAN の紹介](https://www.renom.jp/ja/notebooks/tutorial/generative-model/anoGAN/notebook.html)
* [Keras AnoGAN で異常検知をやってみる](http://cedro3.com/ai/keras-anogan-anomaly/)
* [異常検知とGAN: AnoGan](https://www.slideshare.net/Koichirotamura2/gan-anogan)
* [論文読み Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://aotamasaki.hatenablog.com/entry/2018/04/14/212948)
* [論文まとめ：Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://qiita.com/masataka46/items/09e4e13be097a68174cd)


#### ADGAN

* Paper: [Anomaly Detection with Generative Adversarial Networks](https://openreview.net/forum?id=S1EfylZ0Z)
* [異常検知とGAN: ADGAN](https://www.slideshare.net/Koichirotamura2/gan-adgan)
* [論文まとめ：ANOMALY DETECTION WITH GENERATIVE ADVERSARIAL NETWORKS](https://qiita.com/masataka46/items/4f7f770250fecc911bbc)


#### Efficient GAN

* Paper: [Efficient GAN-Based Anomaly Detection](https://arxiv.org/abs/1802.06222)
* Code: [houssamzenati/Efficient-GAN-Anomaly-Detection](https://github.com/houssamzenati/Efficient-GAN-Anomaly-Detection)
* [論文まとめ＆実装例：EFFICIENT GAN-BASED ANOMALY DETECTION](https://qiita.com/masataka46/items/49dba2790fa59c29126b)
* [異常検知とGAN: Efficient GAN](https://www.slideshare.net/Koichirotamura2/gan-efficient-gan)


#### ALOCC

* Paper: [Adversarially Learned One-Class Classifier for Novelty Detection](https://arxiv.org/abs/1802.09088)
* Code: [khalooei/ALOCC-CVPR2018](https://github.com/khalooei/ALOCC-CVPR2018)
* [ディープラーニングによる異常検知手法ALOCCを実装した](https://qiita.com/kzkadc/items/334c3d85c2acab38f105)
* [論文まとめ＆実装例：Adversarially Learned One-Class Classifier for Novelty Detection](https://qiita.com/masataka46/items/b167a89c11061eee607d)
* [ALOCCを使った「文字画像」を判別する試み](https://devblog.thebase.in/entry/2018/11/07/111126)
* [論文紹介：Adversarially Learned One-Class Classifier for Novelty Detection](https://www.slideshare.net/KazukiAdachi/adversarially-learned-oneclass-classifier-for-novelty-detection)


#### AVID

* Paper: [AVID: Adversarial Visual Irregularity Detection](https://arxiv.org/abs/1805.09521)
* [[DL輪読会]AVID:Adversarial Visual Irregularity Detection](https://www.slideshare.net/DeepLearningJP2016/dlavidadversarial-visual-irregularity-detection)
* [[DL Hacks]AVID: Adversarial Visual Irregularity Detection](https://www.slideshare.net/DeepLearningJP2016/dl-hacksavid-adversarial-visual-irregularity-detection)


#### DML (Deep Metric Learning)

* [Metric Learning for Novelty and Anomaly Detection](https://arxiv.org/abs/1808.05492)
* [Triplet lossを使った高速な異常検知](https://qiita.com/koshian2/items/b4c4ffda99c07a1ac6b8)



### A Survey

* Paper: [Deep Learning for Anomaly Detection: A Survey](https://arxiv.org/abs/1901.03407)
* [【異常検知】Deep Learning for Anomaly Detection: A Survey を読んだ (前編)](https://fisproject.jp/2019/03/deep-learning-for-anomaly-detection-1/)
* [【異常検知】Deep Learning for Anomaly Detection: A Survey を読んだ (後編)](https://fisproject.jp/2019/03/deep-learning-for-anomaly-detection-2/)
* [[DL輪読会]Deep Learning for Anomaly Detection: A Survey](https://www.slideshare.net/DeepLearningJP2016/dldeep-learning-for-anomaly-detection-a-survey-131436499)
* [Deep Learning: A Survey of Surveys](https://qiita.com/bukei_student/items/5d0ec0e7cc36d211e17a)

