from pathlib import Path

import nbformat as nbf

def create_cagmon_tutorial():
    nb = nbf.v4.new_notebook()

    # --- Title and Intro ---
    nb.cells.append(nbf.v4.new_markdown_cell("# Multi-Channel Noise Diagnostics with GWexpy: A CAGMon-Inspired Approach\n\n"
        "このチュートリアルでは、**CAGMon** (Jung et al. 2022) の核心的なアイデアを **GWexpy** で体験します。\n\n"
        "### なぜマルチチャネル解析が必要か？\n"
        "重力波検出器は数千もの補助チャネル（環境モニター、装置ステータス）を持っています。メインの GW チャネルに混入するノイズの原因を特定し、その物理的な性質（線形か非線形か）を理解することは、感度向上のために不可欠です。\n\n"
        "### 本チュートリアルのポイント\n"
        "1.  **物理モデルに基づくノイズ生成**: `gwinc`, `obspy`, `schumann_resonance` を使用し、リアルなデータを合成します。\n"
        "2.  **線形 vs 非線形カップリングの識別**: Pearson 相関係数 (PCC) と Maximal Information Coefficient (MIC) の組み合わせで、カップリングの性質を判別します。\n"
        "3.  **因果関係の推定**: Granger 因果性テストを用いて、ノイズの流れの方向性を確認します。"))

    # --- Theory ---
    nb.cells.append(nbf.v4.new_markdown_cell("## CAGMon で用いられる相関統計量\n\nCAGMon では、線形・非線形を問わずカップリングを特定するために複数の統計量を組み合わせて評価します。\n\n"
        "### 1. Pearson Correlation Coefficient (PCC)\n"
        "線形相関の強度を測る最も一般的な指標です。$-1 \\le r \\le 1$ の値をとり、非線形な関係（例: 二乗の関係）に対しては感度が低くなります。\n\n"
        "$$ r = \\frac{\\sum_{i=1}^n (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum_{i=1}^n (x_i - \\bar{x})^2} \\sqrt{\\sum_{i=1}^n (y_i - \\bar{y})^2}} $$\n\n"
        "### 2. Kendall's Rank Correlation Coefficient ($\\tau$)\n"
        "データ値の大小関係（順位）に注目するノンパラメトリックな指標です。単調な非線形関係にも対応でき、外れ値に強い特徴があります。\n\n"
        "$$ \\tau = \\frac{2}{n(n-1)} \\sum_{i < j} \\text{sgn}(x_i - x_j) \\text{sgn}(y_i - y_j) $$\n\n"
        "### 3. Maximal Information Coefficient (MIC)\n"
        "散布図上に描いたグリッドの相互情報量 $I(X;Y)$ の最大値を求める手法です。**線形・非線形を問わずあらゆる関数関係**に対して高いスコア（$0 \\le MIC \\le 1$）を出力します。\n\n"
        "$$ MIC(x, y) = \\max_{|X||Y| < B} \\frac{I(X; Y)}{\\log_2(\\min(|X|, |Y|))} $$\n\n"
        "CAGMon では、**PCC が低いのに MIC が高い** チャネルを探すことで、従来は見逃されていた非線形ノイズカップリングを特定します。\n\n"
        "### 4. Granger Causality (グレンジャー因果性)\n"
        "2つの時系列データの間に「予測的な因果関係」があるかを判定する手法です。「時間的な遅れ (ラグ) $p$ を考慮したとき、$X$ の過去の値が $Y$ の将来の予測に役立つか」を自己回帰 (AR) モデルを通して検証します。\n\n"
        "1. **制限モデル (Restricted Model)**: $Y$ の過去の値のみで $Y$ を予測\n"
        "$$ Y_t = a_0 + \\sum_{i=1}^p a_i Y_{t-i} + e_t $$\n"
        "2. **非制限モデル (Unrestricted Model)**: $Y$ の過去の値に加えて、$X$ の過去の値も使って $Y$ を予測\n"
        "$$ Y_t = c_0 + \\sum_{i=1}^p c_i Y_{t-i} + \\sum_{j=1}^p d_j X_{t-j} + \\epsilon_t $$\n\n"
        "F検定等を用いて、「係数 $d_j$ がすべてゼロである」という帰無仮説を棄却できた場合、「$X$ は $Y$ にグレンジャー因果性をもつ (ノイズが $X$ から $Y$ に伝搬している可能性がある)」とみなします。"))

    # --- Setup ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Setup & Imports\n\nまずは必要なライブラリをインポートし、GWexpy を初期化します。"))
    nb.cells.append(nbf.v4.new_code_cell("import numpy as np\nimport matplotlib.pyplot as plt\nimport gwexpy\nfrom gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix\nfrom gwexpy.noise import asd, wave\nfrom gwexpy.analysis.stat_info import association_edges, build_graph\n\n# 可視化スタイルの設定\nplt.rcParams['figure.figsize'] = (10, 6)\nplt.rcParams['font.size'] = 12"))

    # --- Data Generation ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. 合成データの生成\n\n物理的なノイズモデルから 5 つのチャネルを生成します。\n\n"
        "| チャネル | ノイズ源 | 性質 |\n"
        "|:---|:---|:---|\n"
        "| **DARM** | aLIGO 検出器モデル | メイン GW チャネル |\n"
        "| **ACC_X** | 地震ノイズ (NHNM 加速度) | **線形結合** |\n"
        "| **MIC_1** | 超低周波音圧 (IDCH) | **非線形（二乗）結合** |\n"
        "| **MAG_Y** | シューマン共鳴 | **無相関** |\n"
        "| **TEMP** | 1日周期 sine + WN | **弱い線形結合** |"))
    
    nb.cells.append(nbf.v4.new_code_cell("from astropy import units as u\n\nfs = 1024  # サンプリングレート [Hz]\ndur = 32   # 長さ [s]\nfreqs = np.arange(1, fs//2 + 1, 1.0)\n\n# 1. ASD (Amplitude Spectral Density) の取得\nasd_darm = asd.from_pygwinc('aLIGO', quantity='darm', fmin=1, fmax=fs//2, df=1)\nasd_acc  = asd.from_obspy('NHNM', quantity='acceleration', frequencies=freqs)\nasd_mic  = asd.from_obspy('IDCH', frequencies=freqs)  # 超低周波音圧ノイズ\nasd_mag  = asd.schumann_resonance(frequencies=freqs)\n\n# 2. ASD から時系列波形を生成\ndarm_base = wave.from_asd(asd_darm, duration=dur, sample_rate=fs, name='DARM_base')\nacc_x     = wave.from_asd(asd_acc,  duration=dur, sample_rate=fs, name='ACC_X')\nmic_raw   = wave.from_asd(asd_mic,  duration=dur, sample_rate=fs, name='MIC_raw')\nmag_y     = wave.from_asd(asd_mag,  duration=dur, sample_rate=fs, name='MAG_Y')\n\n# 3. 温度チャネル (1日周期の正弦波 + ホワイトノイズ)\n# 単位を °C と明示します\ntemp_sine = wave.sine(dur, fs, frequency=1/86400, amplitude=1.0, unit='deg_C', name='TEMP_base')\ntemp      = temp_sine + wave.white_noise(dur, fs, amplitude=0.01, unit='deg_C')\ntemp.name = 'TEMP'\n\n# 4. カップリングの注入 (DARM = base + Linear(ACC) + NonLinear(MIC) + WeakLinear(TEMP))\n# 単位を合わせて加算します\nalpha = 0.5 * u.Unit('s^2')           # [m] / [m/s^2]\nbeta  = 0.1 * u.Unit('m/Pa^2')        # [m] / [Pa^2]\ngamma = 0.005 * u.Unit('m/deg_C')     # [m] / [deg_C]\n\ndarm = darm_base + alpha * acc_x + beta * (mic_raw ** 2) + gamma * temp\ndarm.name = 'DARM'\nmic_1 = mic_raw\nmic_1.name = 'MIC_1'\n\n# TimeSeriesDict にまとめる\ndata = TimeSeriesDict({'DARM': darm, 'ACC_X': acc_x, 'MIC_1': mic_1, 'MAG_Y': mag_y, 'TEMP': temp})\nprint('Generated 5 channels.')"))

    # --- Visualization ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. 可視化と基本統計量\n\n生成されたデータをプロットしてみます。"))
    nb.cells.append(nbf.v4.new_code_cell("data.plot(separate=True, geometry=(5, 1), figsize=(12, 12), sharex=True)\nplt.tight_layout()"))
    nb.cells.append(nbf.v4.new_markdown_cell("周波数領域でも確認します。DARM にどの成分が寄与しているか ASD で見比べます。"))
    nb.cells.append(nbf.v4.new_code_cell("plt.figure()\nplt.loglog(darm.asd(), label='DARM (Total)', color='black', alpha=0.8)\nplt.loglog(darm_base.asd(), label='DARM base', linestyle='--', alpha=0.5)\nplt.loglog((alpha * acc_x).asd(), label='ACC_X contribution (Linear)', alpha=0.7)\nplt.loglog((beta * (mic_raw**2)).asd(), label='MIC_1 contribution (Non-linear)', alpha=0.7)\nplt.xlabel('Frequency [Hz]')\nplt.ylabel('ASD [m/rtHz]')\nplt.legend()\nplt.grid(True, which='both', linestyle=':')\nplt.title('Simulated Noise Budget')"))

    # --- Pairwise Correlation ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 4. ペアワイズ相関解析 (Pearson vs MIC)\n\nCAGMon の核心部です。DARM と各補助チャネルの相関係数を計算し、カップリングの種類を特定します。\n\n"
        "- **PCC (Pearson)**: 線形な相関を捉える。\n"
        "- **MIC (Maximal Information)**: 線形・非線形を問わない依存関係を捉える。"))
    
    nb.cells.append(nbf.v4.new_code_cell("channels = ['ACC_X', 'MIC_1', 'MAG_Y', 'TEMP']\nresults = []\n\nfor ch in channels:\n    target = data[ch]\n    pcc = darm.pcc(target)\n    mic = darm.mic(target)\n    results.append({'channel': ch, 'PCC': pcc, 'MIC': mic})\n\nimport pandas as pd\ndf_res = pd.DataFrame(results)\n\n# 判別ロジックの可視化\ndef categorize(row):\n    if row['MIC'] < 0.2: return 'Uncorrelated'\n    if abs(row['PCC']) > 0.8 * row['MIC']: return 'Linear'\n    return 'Non-Linear'\n\ndf_res['Relation'] = df_res.apply(categorize, axis=1)\ndisplay(df_res)"))

    nb.cells.append(nbf.v4.new_markdown_cell("**結果の考察**:\n"
        "- `ACC_X`: PCC と MIC が共に高く、**線形**と判定されます。\n"
        "- `MIC_1`: **PCC は低いのに MIC は高い** → 二乗カップリングという**非線形**性が正しく抽出されました！\n"
        "- `MAG_Y`: 両方低く、**無相関**です。\n"
        "- `TEMP`: 弱いながらも線形な傾向が見えます。"))

    # --- Batch Analysis ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 5. TimeSeriesMatrix による一括解析とグラフ化\n\n多チャネルを扱う場合は `TimeSeriesMatrix` が便利です。"))
    nb.cells.append(nbf.v4.new_code_cell("matrix = data.to_matrix()\n\n# MIC を使った一括スコアリング\nedges = association_edges(darm, matrix, method='mic', threshold=0.1)\ndisplay(edges[['target', 'score']])\n\n# ネットワーク図の構築 (重要ノードの可視化)\ng = build_graph(edges, weight='score', backend='networkx')\n\nimport networkx as nx\nplt.figure(figsize=(8, 6))\npos = nx.spring_layout(g, seed=42)\nnx.draw(g, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold')\nedge_labels = { (u, v): f\"{d['weight']:.2f}\" for u, v, d in g.edges(data=True) }\nnx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)\nplt.title('Coupling Network (MIC scores)')"))

    # --- Granger Causality ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 6. 因果関係の推定 (Granger Causality)\n\n相関は「つながり」を示しますが、どちらが原因かは分かりません。グレンジャー因果性を用いて方向性を確認します。"))
    nb.cells.append(nbf.v4.new_code_cell("# DARM が ACC_X によって引き起こされているか？\nres1 = darm.granger_causality(acc_x, maxlag=5)\nprint(f\"ACC_X -> DARM: p-value = {res1.min_p_value:.2e} (Lag {res1.best_lag})\")\n\n# 逆方向 (DARM -> ACC_X) は？\nres2 = acc_x.granger_causality(darm, maxlag=5)\nprint(f\"DARM -> ACC_X: p-value = {res2.min_p_value:.2f}\")\n\nif res1.min_p_value < 0.05:\n    print('Conclusion: ACC_X Granger-causes DARM.')"))

    # --- Conclusion ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 7. まとめ\n\nこのチュートリアルでは、以下の GWexpy の機能を活用しました：\n- **物理ベースのノイズ生成**: `gwinc`, `obspy` などから ASD を取得し時系列化。\n- **高度な相関指標**: PCC に加え、非線形カップリングに強い **MIC** を活用。\n- **バッチ解析 & グラフ化**: `TimeSeriesMatrix` と `association_edges` による効率的な多チャネル処理。\n- **因果関係**: `granger_causality` によるノイズ伝搬方向の特定。\n\nこれにより、CAGMon のような高度なノイズ診断ワークフローが、GWexpy を通じて容易に構築できることが分かりました。"))

    # --- Save ---
    repo_root = Path(__file__).parent.parent.parent
    tutorial_path = repo_root / 'examples' / 'case-studies' / 'case_cagmon_noise_diagnostics.ipynb'
    tutorial_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tutorial_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f'Successfully created {tutorial_path}')

if __name__ == '__main__':
    create_cagmon_tutorial()
