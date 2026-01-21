#!/usr/bin/env python
"""Field4D 描画デモスクリプト（ヘッドレス環境対応）

CIや画像出力用の最小限のプロットスクリプト。
Matplotlib の Agg バックエンドを使用し、GUIなしで画像を生成します。

使用例:
    python plot_field4d_demo.py  # -> field4d_demo.png を出力
"""

import matplotlib

matplotlib.use('Agg')  # ヘッドレスバックエンド

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from gwexpy.types import Field4D


def create_sample_field():
    """再現性のあるサンプルField4Dを生成."""
    np.random.seed(42)

    # 軸定義
    nt, nx, ny, _ = 10, 32, 32, 1
    t = np.linspace(0, 1, nt) * u.s
    x = np.linspace(-5, 5, nx) * u.m
    y = np.linspace(-5, 5, ny) * u.m
    z = np.array([0]) * u.m

    # 移動ガウシアン
    T, X, Y = np.meshgrid(t.value, x.value, y.value, indexing='ij')
    x_center = T * 5  # 5 m/s
    data = 10.0 * np.exp(-((X - x_center)**2 + Y**2) / 2)
    data = data[:, :, :, np.newaxis]

    return Field4D(
        data,
        unit=u.V,
        axis0=t,
        axis1=x,
        axis2=y,
        axis3=z,
        axis_names=['t', 'x', 'y', 'z'],
        axis0_domain='time',
        space_domain='real'
    )


def main():
    """メイン関数: 4パネルの概要図を生成."""
    field = create_sample_field()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel 1: XY断面
    field.plot_map2d('xy', at={'t': 0.5 * u.s}, ax=axes[0, 0], add_colorbar=True)
    axes[0, 0].set_title('XY Map (t=0.5s)')

    # Panel 2: 時系列
    points = [
        (0.0 * u.m, 0.0 * u.m, 0.0 * u.m),
        (2.5 * u.m, 0.0 * u.m, 0.0 * u.m),
    ]
    field.plot_timeseries_points(points, ax=axes[0, 1])
    axes[0, 1].set_title('Time Series')
    axes[0, 1].legend(fontsize=8)

    # Panel 3: Xプロファイル
    field.plot_profile('x', at={'t': 0.5 * u.s, 'y': 0.0 * u.m, 'z': 0.0 * u.m}, ax=axes[1, 0])
    axes[1, 0].set_title('X Profile (t=0.5s, y=0)')

    # Panel 4: 時間-空間マップ
    field.plot_time_space_map('x', at={'y': 0.0 * u.m}, ax=axes[1, 1], add_colorbar=True)
    axes[1, 1].set_title('Time-Space Map')

    plt.tight_layout()
    output_path = 'field4d_demo.png'
    fig.savefig(output_path, dpi=150)
    print(f'Saved: {output_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
