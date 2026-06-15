# Map distortion fix (GIS-based) / GISによる地図ゆがみ補正

バカ世界地図ベースの HOI4 マップ（`bakasekai/map/`）について、**作風（どの国・プロヴィンスがどこにあるか）は保ったまま、局所的な幾何ゆがみ（縦横比のズレ）を実地理（GIS）に合わせて補正する**ための診断＆補正ツール群です。

## TL;DR

- マップは **5120×2560 = 2:1**。正距円筒図法（経度360°×緯度180°）として **全体の縦横比は正しい**。
- ただし大陸・地域ごとに **手描き由来の局所的なゆがみ／位置ズレ**があり、実地理と比べると陸海が **約16.8%** 食い違う。
- 全レイヤーを **同一の変位場でワープ**（プロヴィンスは最近傍補間で色＝ID保持）すると、食い違いを **約6.8% まで低減**できた（概念実証）。
- この方式なら **province ID が変わらない**ので、`definition.csv` / states / history / supply などの既存データは壊れない。

## なぜ「ワープ」なのか

プロヴィンスをゼロから引き直すと全 province ID が変わり、mod 全体（`definition.csv`・states・history・supply・buildings）が連鎖的に壊れる。
代わりに **既存ラスターを同じ変換で歪ませる（ワープ）** ことで、各プロヴィンスの固有色＝IDを保ったまま「形と位置だけ」直せる。

- `provinces.bmp` / `terrain.bmp` / `rivers.bmp` … **最近傍補間**（色・インデックスを混色させない＝無効色を作らない）
- `heightmap.bmp` / `world_normal.bmp` … **バイリニア補間**

## スクリプト

| ファイル | 役割 |
|---|---|
| `diagnose_aspect.py` | 現行マップの陸海マスクと実地理（Natural Earth）を同一フレームで比較し、**どこがどれだけズレているか**を画像＋数値で可視化 |
| `warp_map.py` | 陸海マスクのブロックマッチングで変位場を推定し、全レイヤーをワープ。**保護領域(`protect.csv`)と制御点(`gcps.csv`)で人手調整可能**。before/after を計測。`--apply` で全解像度の補正レイヤーを `out/corrected/` に出力 |
| `gcps.csv` | 制御点(GCP)の定義ファイル（テンプレート、既定は全コメントアウト） |
| `protect.csv` | 保護領域の定義ファイル（テンプレート、既定は全コメントアウト） |

## 人手調整：保護領域と制御点（GCP）

自動ワープは「全てのゆがみは実地理に対する誤差」と仮定するため、そのままでは
(1) 意図的なバカ歪みも実地理へ寄せてしまう、(2) 自動探索範囲を超える大きなズレ（豪州など）は直しきれない、という限界がある。これを次の2つで補う。

### 保護領域 `protect.csv`（意図的な歪みを守る）

正規化座標（0,0=左上, 1,1=右下）の矩形で「**ワープしない領域**」を指定する。
縁はフェザリングされるので周囲との継ぎ目は出ない。

```csv
# columns: name,x0,y0,x1,y1
fictional_island,0.40,0.30,0.46,0.36
```

PNGマスク（白=保護）でも指定可：`--protect-mask mask.png`

### 制御点 `gcps.csv`（大きなズレを手当て）

「この**現在地(src)** に描かれている地形は、本当は**この位置(dst)** にある」という対応点を与えると、その点付近だけを局所的に（ガウス減衰、既定半径=フレーム幅の6%）動かす。遠方には影響しない安全設計。

```csv
# columns: name,src_x,src_y,dst_x,dst_y   (すべて正規化0..1)
# dst は実 lon/lat から:  dst_x=(lon+180)/360,  dst_y=(90-lat)/180
australia_centre,0.835,0.655,0.872,0.639
```

- src は `out/overlay.png`（または HOI4 nudge ツール）で読み取る。
- 影響半径は `--gcp-radius`（既定0.06）で調整。
- **GCPは正確に。** 不正確な点はその周辺を逆に悪化させる（局所影響なので被害は限定的）。
- 配置確認用に `out/control_overlay.png`（赤線=src→dst、黄=保護領域）を出力する。

### GISデータの取得

Natural Earth 公式CDN（`naciscdn.org`）はこの環境ではブロックされているため、GitHub ミラーから取得する：

```bash
mkdir -p /tmp/gis && cd /tmp/gis
base="https://github.com/nvkelso/natural-earth-vector/raw/master/110m_physical"
for ext in shp shx dbf prj; do curl -sSL -o ne_110m_land.$ext "$base/ne_110m_land.$ext"; done
```

### 実行

```bash
pip install -r requirements.txt
export GIS_DIR=/tmp/gis

python3 diagnose_aspect.py --width 1280     # 診断（out/ に overlay.png, diff.png 等）
python3 warp_map.py        --work 1024      # 補正の概念実証（out/ に diff_before/after, プレビュー）
python3 warp_map.py --gcps gcps.csv --protect protect.csv   # 手動調整を反映
python3 warp_map.py        --apply          # 全解像度の補正レイヤーを out/corrected/ に生成
```

## 出力（`out/`）

- `overlay.png` … 現行 terrain に実地理の海岸線（赤）を重ねたもの
- `diff.png` / `diff_before.png` / `diff_after.png` … 陸海の不一致。**灰=一致 / 赤=マップが陸だが実際は海 / 青=実際は陸だがマップは海**。海岸沿いの赤青ペア＝位置ズレの痕跡
- `field_magnitude.png` … 推定された変位の大きさ（明=大きく動かす）
- `terrain_corrected_preview.png` … ワープ後の terrain プレビュー
- `control_overlay.png` … 制御点（赤線 src→dst）と保護領域（黄）の配置確認
- `control_overlay_demo.png` … 上記機能のデモ（豪州GCP＋北アフリカ保護領域の例）

## 計測結果（概念実証 @1024×512）

| | map-only（過剰な陸） | gis-only（欠けた陸） | 合計 |
|---|---|---|---|
| before | 9.27% | 7.51% | **16.77%** |
| after  | 4.33% | 2.44% | **6.77%** |

## 重要な注意・限界（本番適用の前に）

1. **これは概念実証であり、まだ本番適用していない。** `warp_map.py` は既定では `out/` にプレビューを書くだけで、`bakasekai/map/` 本体は変更しない。
2. **自動ワープは「全てのゆがみは実地理に対する誤差」と仮定する。** 意図的なバカ世界地図デザインのゆがみがあれば、それも実地理へ寄せてしまう。意図的なゆがみは GCP（制御点）から除外する／保護領域を設けるなどの対応が必要。
3. **大きすぎる位置ズレは未補正。** 例：オーストラリアは探索半径を超えてズレており、単一スケールでは直りきらない。→ 粗→密のマルチスケール探索が必要（今後の課題）。
4. **プロヴィンスの整合性検証が必須。** ワープで 1 プロヴィンスが分断・消失していないか（連結成分チェック、`definition.csv` の全IDが残っているか）を `--apply` 後に必ず検証すること。
5. **派生データの再生成が必要。** `positions.txt`・`unitstacks.txt`・`buildings.txt` の座標・`adjacencies.csv`・supply nodes 等は形が変わると合わなくなるため、HOI4 の nudge ツール等で再生成・再確認する。
6. 陸海マスクは `heightmap.bmp`（海面=値71）から近似生成している。より厳密には `definition.csv` の sea/land 分類＋`provinces.bmp` から作る方が正確。

## 今後の進め方（提案）

1. ~~意図的ゆがみの保護領域指定~~ → 実装済み（`protect.csv` / `--protect-mask`）。
2. ~~大きな位置ズレの手当て~~ → 制御点で対応可（`gcps.csv`）。さらに自動化するならマルチスケール（粗→密）ブロックマッチングを追加。
3. `--apply` 出力に対するプロヴィンス連結性・ID 残存の自動検証スクリプト追加。
4. 検証OK後に `bakasekai/map/` へ反映し、HOI4 で読み込み確認（nudge で派生データ再生成）。
