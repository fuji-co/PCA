#  主成分分析_逆変換
#  https://www.yakupro.info/entry/ml-pca

"""
寄与率(%): pca.explained_variance_ratio_             //Contribution Ratio
固有値(主成分の分散): pca.explained_variance_          //Eigen Value
固有ベクトル(主成分の方向): pca.components_             //Eigen Vector
主成分スコア(データの主成分方向での値): pca.transform(X)  //Principal Component Score
"""

from typing import Dict
import pandas as pd
from sklearn.decomposition import PCA  # 主成分分析器
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def main():
    # 解析準備
    project_dir = Path(__file__).parent
    data_dir = project_dir / 'data/'
    out_dir = project_dir / 'results/'
    out_dir.mkdir(exist_ok=True, parents=True)

    input_file = data_dir / 'raw_score.csv'  # 解析データのファイル名
    out_files = {
        "元データ": out_dir / "00_raw_score.csv",
        '標準化データ': out_dir / "01_standard_score.csv",
        '寄与率': out_dir / "02_contribution_ratio.csv",
        '固有ベクトル': out_dir / "03_eigen_vector.csv",
        '主成分スコア': out_dir / '04_std_principal_score.csv',
        '逆変換': out_dir / '05_std_recovered.csv',
        '逆変換元データ': out_dir / '06_recovered_raw_score.csv'
    }

    component_number = 2  # 主成分の数を指定

    # 解析データの形式
    header_flg = 0  # ヘッダー（列名）は必要：　有り 0, （無し None）
    index_flg = 0  # index（Sample名)：　有り 0,無し None

    # 元データ読み込み
    raw_score = pd.read_csv(input_file, header=header_flg, index_col=index_flg)

    # 解析の実行
    pca_analysis(raw_score, component_number, out_files)


def pca_analysis(raw_score, component_number, out_files):
    assert set(out_files.keys()) == {'元データ', '標準化データ', '寄与率', '固有ベクトル', '主成分スコア', '逆変換', '逆変換元データ'}

    # 元データの出力
    data_output("元データ", raw_score, out_files)

    # 標準化
    sc = StandardScaler().fit(raw_score)
    std_score = pd.DataFrame(sc.transform(raw_score))
    data_output("標準化データ", std_score, out_files)

    # 主成分分析の実行
    pca = PCA(n_components=component_number)
    pca.fit(std_score)

    # 各種データ出力
    explained_variance_ratio = pd.DataFrame(pca.explained_variance_ratio_)
    components = pd.DataFrame(pca.components_)
    list_std_principal_score = pd.DataFrame(pca.transform(std_score))

    data_output("寄与率", explained_variance_ratio, out_files)
    data_output("固有ベクトル", components, out_files)
    data_output("主成分スコア", list_std_principal_score, out_files)

    # 逆変換
    list_std_recovered_score = pca.inverse_transform(list_std_principal_score)
    std_recovered_score = pd.DataFrame(list_std_recovered_score, columns=raw_score.columns)
    data_output("逆変換", std_recovered_score, out_files)

    # 標準化データを元にもどす
    recovered_raw_score = pd.DataFrame(sc.inverse_transform(std_recovered_score),
                                       index=raw_score.index, columns=raw_score.columns)
    data_output("逆変換元データ", recovered_raw_score, out_files)
    return


def data_output(name, df, files: Dict[str, Path]):  # 結果の出力
    show_data(df, name)
    dump(df, files[name])


def show_data(df: pd.DataFrame, name: str):
    print("\n", name, "\n", df)


def dump(df, file: Path):
    df.to_csv(file)


if __name__ == '__main__':
    main()
