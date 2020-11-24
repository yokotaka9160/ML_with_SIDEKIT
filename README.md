# 音声データから音響特徴量の抽出用プログラム
- ライブラリ：SIDEKIT
- 抽出特徴量：MFCC , PLP , フォルマント
- 説明：MFCC,PLPはHDF5形式で出力され、hdf5TOcsvで差分や変化量などの特徴量が加えられた形のcsv形式で出力される
- 補足：MFCC,PLPはPython・フォルマントはMATLABのLPC分析
