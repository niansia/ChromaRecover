# ChromaRecover

[English](README.md) | [繁體中文](README.zh-TW.md) | [简体中文](README.zh-CN.md)

![Python 3.10–3.13](https://img.shields.io/badge/Python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white)
![Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-3DA639)
![Experimental Alpha](https://img.shields.io/badge/Status-Experimental%20Alpha-E69500)
![Local-first](https://img.shields.io/badge/Privacy-Local--first-5B5BD6)
[![CI](https://github.com/niansia/ChromaRecover/actions/workflows/ci.yml/badge.svg)](https://github.com/niansia/ChromaRecover/actions/workflows/ci.yml)

ChromaRecover 是一套本機優先的 Python 工具，用來復原並呈現由細微色彩差異承載的空間結構。它回傳 top-k 遮罩、疊圖、品質警告，以及明確的 `ok`、`uncertain` 或 `retry_recommended` 狀態，不會被迫猜出文字答案。

> Alpha 範圍：digital mode 是目前穩定的基線。實驗性 Camera Alpha 提供保守的文件透視校正／展平、紙張與螢幕拍攝設定、多幀融合、反光無效區標記、輕度復原假設，以及證據回映射到原始照片。`mode="auto"` 會辨識明顯的螢幕摩爾紋，否則回退到 digital。紙張照片在取得獨立裝置留出校準前，請明確選擇 `camera-paper`。目前信心度採保守設計，尚未以獨立裝置資料完成校準。

| 直方圖匹配的馬賽克 | 連續證據 | 可稽核疊圖 |
| --- | --- | --- |
| ![產生的多邊形馬賽克](examples/digital/demo_input.png) | ![復原的連續證據](examples/digital/demo_result/evidence_01.png) | ![復原疊圖](examples/digital/demo_result/overlay_01.png) |

這個由專案自行產生的案例，將 `820` 隱藏在某一種多邊形顏色的局部密度中，並讓隨機負例具有相同數量的特殊色彩元件。證據圖會把空間訊號顯示出來；此決定性案例回傳 `ok`，而存在多個同樣合理遮罩的案例會維持 `uncertain`。這是 evidence-first 的預期行為，不是 OCR 承諾。

## 系統架構

![ChromaRecover evidence-first 系統架構](docs/assets/chromarecover-architecture.png)

復原流程與持續紅隊／藍隊驗證迴路整合在同一張圖中：輸入安全、拍攝模型、多族群證據假設、保守決策及可稽核產物。圖中的物件皆可在[可編輯 PowerPoint 原始檔](docs/assets/ChromaRecover-research-architecture.pptx)中個別調整。

## 為什麼使用 ChromaRecover？

有些結構主要由相對色彩而非強烈亮度邊緣編碼，因此人眼、OCR 與一般視覺模型都可能難以可靠判讀。ChromaRecover 會先揭露視覺證據，再嘗試賦予標籤：

- 先復原結構，再進行 OCR 或數字解讀；
- 在本機 CPU 執行，不上傳、不需要帳號，也沒有遙測；
- 證據不足時回傳不確定，不憑空補畫或猜答案。

**如果證據不足，ChromaRecover 會明確說明。**

## 安裝

在第一個套件註冊表版本發布前，請下載或 clone 本 repository，進入根目錄後安裝已追蹤的原始碼：

```
python -m pip install .
```

若要執行測試與貢獻者工具：

```
python -m pip install ".[dev]"
```

Windows Python 3.10 在 repository 路徑含非 ASCII 字元時，可能觸發上游 editable-install `.pth` 編碼問題。上面的非 editable 安裝方式較可靠；開發環境建議使用 Python 3.11 以上。

## Python API

```python
from chromarecover import recover

result = recover("image.png", mode="digital", top_k=3)
result.save("result")
if result.best is None:
    print(result.status, "沒有可辯護的候選")
else:
    print(result.status, result.best.decision_confidence)
```

`result.best` 的型別刻意設為 `Candidate | None`：空白、近乎單色或資訊量過低的影像，會正常回傳沒有候選的 `uncertain`，而不是捏造結果。

數字解讀是選用功能，而且只會在視覺證據存在後執行：

```python
result = recover("plate.png", semantics="auto")
print(result.semantics)  # 假設、信心度、接受理由、候選間一致性
```

它結合多字型形狀距離、孔洞／拓樸與投影式字形切割。只有通過接受條件的讀值會出現在公開 `hypothesis` 欄位；未接受的模板標籤只保留為候選內的字元層級診斷資訊。數字假設不會改動遮罩，也不會把缺失像素變成證據。

當三個候選遮罩、至少兩個獨立視覺證據族群高度一致時，即使復原狀態仍是 `uncertain`，也可能接受可讀的數字序列。這只驗證形狀，不能提升視覺狀態、重排候選或宣稱復原缺失像素。

### 相機照片

相機模式預設會自動尋找可信四邊形。若文件邊界不明確，請傳入裁切範圍或原始影像座標中的四個角點：

```python
result = recover("photo.jpg", mode="camera", roi=(120, 80, 1880, 1320))
result = recover(
    "photo.jpg",
    mode="camera",
    document_corners=[[130, 92], [1872, 65], [1904, 1318], [105, 1340]],
)
```

角點必須是四個互異、有限且能形成可用凸四邊形的點。系統容許 10% 的小幅座標邊界，供次像素或略微超框估計使用；非有限值、重複點、退化或距離過遠的點，會在透視轉換前拋出 `InvalidInputError`。

皺褶與光照影響明顯時使用 `camera-paper`；更新帶紋與摩爾紋明顯時使用 `camera-screen`。一般 `camera` 會依量測證據選擇設定。自動展平只使用實際觀測到的頁面邊界，幾何不可信時會棄權。

若反光或遮擋在多張照片間移動，可用 burst fusion 從其他幀中真正被觀測到的像素取得證據：

```python
from chromarecover import recover_burst

result = recover_burst(
    ["frame-1.jpg", "frame-2.jpg", "frame-3.jpg"],
    mode="camera-screen",
    return_debug=True,
)
```

相機校正只估計干擾場，不宣稱能從單張照片復原遭截斷、遮擋或嚴重失焦的資訊。請檢查 `result.quality.warnings`，並在系統建議時重新拍攝。

`ok` 一律使用 `best.decision_confidence` 所代表的同一個公開信心度；證據族群閘門不能用另一個分數跨過狀態門檻。相機與 auto 狀態在取得具獨立授權、依來源群組留出的拍攝資料前，仍屬未校準狀態。

`ok` 表示系統找到符合決策契約的一致色彩／空間結構；它**不代表**一般照片一定含有刻意隱藏的訊息，也不保證語意解讀正確。

## CLI

```
chromarecover image.png --out result --top-k 3 --semantics auto
chromarecover photo.jpg --mode camera --roi 120,80,1880,1320 --out result
chromarecover --burst one.jpg two.jpg three.jpg --mode camera-screen --debug --out result
```

輸出目錄會區分三種意義：`mask_01.png` 是選定的色彩支撐類別，`structure_01.png` 是其空間包絡，`evidence_01.png` 是連續局部支撐；`overlay_01.png` 同時呈現前兩者。schema `0.5` 的 `result.json` 記錄轉換、品質與來源。全解析度的支撐及結構遮罩保持在原始影像座標。

為避免手機影像造成記憶體尖峰，連續證據與疊圖限制在最長邊 2048 像素的展示空間，並在 `artifact_spaces` 記錄尺寸與縮放比例。`--debug` 另會儲存校正後觀測、無效區域、primitive 標籤與復原假設。

信心度也分層處理：`structure_confidence` 表示拍攝懲罰前的候選；`capture_confidence` 納入輸入品質；`decision_confidence` 再納入歧義與最終狀態。Alpha 期間，`overall_confidence` 僅是 `decision_confidence` 的相容別名。

## 設計保證

- Mask-first：OCR 或檔名不能決定復原像素。
- Semantics-last：數字辨識只能驗證可讀形狀，不能生成色彩支撐或推翻弱證據棄權。
- 多假設：前景顏色、極性與面積不被寫死。
- 分層相機路徑：幾何、光度干擾估計、原始／校正色彩證據、原生多邊形 primitives、局部分佈圖、圖分組與逆向映射皆可稽核。
- 證據保存：burst fusion 使用配準後的實際觀測；單幀去模糊僅以受限假設分支呈現，不會被重新標示為 ground truth。
- 保守失敗處理：可讀但無法定論的輸入不是例外。
- 本機優先：不上傳、不需要帳號、沒有遙測。
- 可重現：每個結果都儲存決定性 seed 與設定指紋。

詳見[演算法說明](docs/algorithm.md)與[專案契約](PROJECT_SPEC.md)。

## 限制與非目標

本工具不是醫療診斷工具，也不會宣稱能從單張相機照片重建物理上真實的顏色或原始像素。嚴重失焦、截斷、反光、遮擋及取樣不足都可能永久破壞資訊；深度且不可展開的皺摺需要經學習或校準的表面模型，不在這個 CPU Alpha 範圍內。

預設單張影像安全上限為 50 megapixels，可涵蓋 8064×6048 的手機原圖。記憶體有限的系統仍應先裁切目標區域，或使用高品質解碼器只降採樣一次。

檔案系統輸入在解碼前限制為 100 MiB。Burst 會先檢查所有輸入，限制為 2–12 幀及合計 80 megapixels。嵌入式 ICC profile 在 LittleCMS 解析前限制為 4 MiB；過大的 profile 會被忽略並記錄。此本機函式庫不假裝提供安全的 process-internal 解碼逾時；網路服務必須依[安全模型](docs/security-model.md)增加程序隔離、wall-time 與記憶體限制。

Alpha 的 inline typing 採 best-effort。套件包含 `py.typed`，但 OpenCV／NumPy 的完整靜態型別閘門是 v0.4 工作；目前標記不表示所有內部陣列操作都已通過 mypy。

## Roadmap 與貢獻

目前優先事項是具獨立授權的相機拍攝、更多 primitive families、較低記憶體的 burst processing，以及量測導向的效能改善。已完成範圍與發布門檻記錄在[公開 roadmap](docs/roadmap.md)；新功能原則上必須同時附上 regression case 與 hard negative。

適合首次貢獻的小型工作整理在[貢獻者題目](docs/contributor-ideas.md)，其中部分已發布為 [`good first issue`](https://github.com/niansia/ChromaRecover/labels/good%20first%20issue) 與 [`help wanted`](https://github.com/niansia/ChromaRecover/labels/help%20wanted)。

## 開發檢查

```
pytest
ruff check .
python benchmark/evaluate_synthetic.py --assert-thresholds
python benchmark/evaluate_nuisance.py --assert-thresholds
python benchmark/evaluate_mosaic.py --assert-thresholds
```

CI 會測試 Python 3.10–3.13，並涵蓋 NumPy 1.26／OpenCV 4.10 與 NumPy 2.x／OpenCV 4.14。OpenCV 5 與未來 NumPy major release 不在目前支援範圍，必須先通過專門的完整測試與 benchmark 才會放寬版本上限。

目前 golden inventory 包含 20 個決定性生成案例；不宣稱它們是真實相機照片。外部相機幾何會使用具授權的 SmartDoc dataset 另外評估，而且不會提交第三方影格，詳見[外部資料集說明](docs/external-datasets.md)。私人 blueprint 衍生 fixture 不存在於公開 tree 或工具中，也不是 CI 的必要條件。

ChromaRecover 採 Apache-2.0 授權。社群治理將在 Public Alpha 期間持續成熟，並朝 1.0 推進。
