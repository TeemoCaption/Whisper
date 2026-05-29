# Whisper-medium 語言專屬 AdaLoRA 方法

## 目標

本專案固定使用 `openai/whisper-medium` 作為共享底座：

- `zh-TW` 音訊輸出台灣華語文字。
- `nan-tw` 音訊輸出台語文字。

`nan-tw` 不是台語轉華語任務。資料前處理會把 Common Voice 台語文本中的括號台羅或白話字抽到 `romanization_text`，主目標保留台語文字。

## 方法定位

第一篇客語低資源論文的可借鑑點是自適應低秩適應。本專案使用 `AdaLoRA`，讓每個 adapter 內部的低秩容量可依訓練訊號動態分配，而不是固定 rank。

第二篇 LoRA-Whisper 的可借鑑點是共享 Whisper 底座，並用語言專屬 adapter 擴充多語言辨識能力。本專案採用相同擴充方向，但將固定秩 LoRA 改為語言專屬 AdaLoRA：

```text
shared Whisper-medium backbone
+ zh-TW AdaLoRA adapter
+ nan-tw AdaLoRA adapter
```

新增語言時不重新訓練 Whisper 底座，只新增該語言的 adapter 設定與訓練資料篩選。

推論時不採用信心閥值混合 adapter，而是先用 Whisper encoder 的凍結表示訓練一個輕量語言分類頭，預測輸入音訊是 `zh-TW` 或 `nan-tw`，再選擇對應 adapter。分類頭使用注意力池化，讓模型自行加權較有語言辨識線索的時間片段，再接兩層 MLP；這比固定平均整段音訊更能保留有效語音線索。

## 設定方式

低秩方法只保留兩個訓練設定檔：

- `configs/config.yaml`：8GB VRAM 設定。
- `configs/config_h100.yaml`：H100 設定。

語言差異不再複製成多份設定檔，而是在訓練指令指定：

```powershell
python .\scripts\train.py --config .\configs\config.yaml --language zh-TW
python .\scripts\train.py --config .\configs\config.yaml --language nan-tw
```

設定檔中的低秩模式固定為 `adapter_scope: language`，代表每個語言都有自己的 adapter。

Linux / H100：

```bash
python scripts/train.py --config configs/config_h100.yaml --language zh-TW
python scripts/train.py --config configs/config_h100.yaml --language nan-tw
```

`--language` 會同時設定：

- `peft.active_language`
- `data.language_filter`
- 語言專屬輸出資料夾
- 語言專屬 wandb run name

## 建議實驗順序

1. 執行 `scripts/prepare_cv.py` 產生整併 TSV。
2. 執行 `scripts/train_lang_classifier.py --config configs/config.yaml` 訓練語言分類頭。
3. 執行 `scripts/train.py --config configs/config.yaml --language zh-TW` 訓練華語 adapter。
4. 執行 `scripts/train.py --config configs/config.yaml --language nan-tw` 訓練台語 adapter。

主要檢查項目為語言專屬 AdaLoRA 在 `zh-TW`、`nan-tw` 與整體 CER 上的表現。
