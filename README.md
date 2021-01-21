# 音声データから音響特徴量の抽出用プログラム（ALL_datafeature.py/JVPD_formants.m/JVPD_formants.py）
- ライブラリ：SIDEKIT
- 抽出特徴量：MFCC , PLP , フォルマント
- 説明：MFCC,PLPはHDF5形式で出力され、hdf5TOcsvで差分や変化量などの特徴量が加えられた形のcsv形式で出力される
- 補足：MFCC,PLPはPython・フォルマントはMATLABのLPC分析
# 音響特徴量データをMATLAB用に加工するプログラム（hfd5TOcsv.py）
- 説明：上記のプログラムはHDF5形式で抽出するため、CSV仕様に変更する。  
ΔMFCC、ΔPLP、Δフォルマントなど追加の特徴量計算も実装
# SVMの学習曲線及び検証曲線を出力するプログラム（JVPD_rbf_SVM.py）
- ライブラリ：scikitlearnなど
- 説明：最も分類精度の高かったSVMの性能評価を行う。評価指標は学習曲線と検証曲線
