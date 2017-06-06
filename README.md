説明：
==============

1.必要環境：
-------------

1. Python3 (3.4~3.6可)  
2. gcc 5+  
3. Python3 ライブラリー：  
  * numpy: `pip3 install numpy`  
  * pillow: `pip3 install pillow`  

2.使い方：
--------------

* 初期化（始めて使うが時が必要）： `make init`
* 簡単な実行： `python3 recommend.py`

パラメーター:
* **-t**　k-means関数をテストするモード、ランダム二次元データでクラスタリング効果がをテストして、テスト結果を二次元画像にプロットして、'./k_means_test.png'に保存します。
* **-k**　k-meansのk（中心数）を設定する。
* **-u**　レコメンドするユーザーのIDを指定する、範囲は1〜943。

例：
* `python3 recommend.py`:　k=10；　1, 345, 579, 900番のユーザーに映画を推薦する（デフォルト値）
* `python3 recommend.py -t -k 20`:　k=20 時の k-means関数をテストする, 結果が'./k_means_test.png'に保存されます。
* `python3 recommend.py -k 5 -u '(943, 800, 777, 543)'`:　k=5；　943, 800, 777, 543番のユーザーに映画を推薦する（デフォルト値）
* `python3 recommend.py -u '(200)'`:　k=10；　200番のユーザーに映画を推薦する

3.ファイルズの構造：
-------------

├── **data**　 // データセットの置く場所  
│   └── **dataset.tar.gz**　 // データセット  
├── **log**　 // ログファイルが生成される場所  
│   └──  
│  
├── **setting.py**　 // ログ設定などグローバルな設定  
├── **func.py**　 // 主の処理コード  
├── **recommend.py**　 // 実行コード  
│  
├── **makefile**　 // 初期化用（cコードのコンパイルとデータセットの解凍）  
├── **move_point.c**　 // k-meansの中もっと計算量が多い部分、Pythonなら遅すぎますので、ｃで書きました  
└── **report.md**　 // レポートのmarkdownファイル  
