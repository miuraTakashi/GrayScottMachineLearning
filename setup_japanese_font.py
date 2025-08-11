# 日本語フォント設定スクリプト
import matplotlib
import platform
import matplotlib.pyplot as plt

def setup_japanese_font():
    """
    matplotlibで日本語フォントを設定する関数
    """
    # OSに応じて日本語フォントを設定
    if platform.system() == 'Darwin':  # macOS
        matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    elif platform.system() == 'Windows':  # Windows
        matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    else:  # Linux
        matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

    # フォントサイズ設定
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化けを防ぐ
    
    print(f"使用フォント: {matplotlib.rcParams['font.family']}")
    print(f"OS: {platform.system()}")
    
    return True

def test_japanese_font():
    """
    日本語フォントが正しく設定されているかテストする関数
    """
    setup_japanese_font()
    
    # 日本語表示テスト
    plt.figure(figsize=(8, 6))
    plt.plot([1, 2, 3, 4], [1, 4, 2, 3], 'o-', linewidth=2, markersize=8)
    plt.title('日本語フォントテスト - クラスター分析結果', fontsize=14, fontweight='bold')
    plt.xlabel('X軸 (パラメータ)', fontsize=12)
    plt.ylabel('Y軸 (値)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(['テストデータ'], loc='upper left')
    plt.tight_layout()
    plt.show()
    
    print("✅ 日本語フォント設定完了！")

if __name__ == "__main__":
    test_japanese_font() 