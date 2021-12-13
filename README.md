# 主成分分析
## 動作
主成分分析および逆解析を実施  
主成分数は以下の行で設定  
component_number = 2  

## 入力ファイル  
dataフォルダ内に保管  
「raw_score.csv」  

ファイル例
>name,x1, x2, x3, x4  
>a,80, 90, 70, 80  
>b,90, 80, 90, 80  
>c,20, 30, 80, 90  
>d,70, 90, 20, 40  
>e,30, 40, 20, 30  


## 出力ファイル  
resultsフォルダに出力  
「元データ :00_raw_score.csv」  
「標準化データ:01_standard_score.csv」  
「寄与率:02_contribution_ratio.csv」  
「固有ベクトル:03_eigen_vector.csv」  
「主成分スコア:04_std_principal_score.csv」  
「逆変換:05_std_recovered.csv」  
「逆変換元データ:06_recovered_raw_score.csv」  
