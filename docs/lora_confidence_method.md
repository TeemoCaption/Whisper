# Whisper-medium 雙語低秩適應與信心閥值方法對齊

## 目標

本專案固定使用 `openai/whisper-medium` 作為共享底座，目標輸出如下：

- `zh-TW` 音訊輸出台灣華語文字。
- `nan-tw` 音訊輸出台語文字。

`nan-tw` 不是台語轉華語任務。資料前處理會把 Common Voice 台語文本中的括號台羅或白話字抽到輔助欄位，主目標保留台語漢字文字。

## 對齊兩篇論文的核心方法

第一篇客語低資源論文的可借鑑點是使用自適應低秩適應，讓 Whisper 在低資源語音辨識中只訓練少量參數。本專案在 `configs/config.yaml` 預設使用 `whisper_train.peft.method: adalora` 對應這個方向，也保留 `lora` 作為固定秩基線。

第二篇 LoRA-Whisper 的可借鑑點是共享 Whisper 底座、每個語言使用獨立低秩模組，避免多語言資料混合訓練時互相干擾。本專案已在設定檔定義語言專屬轉接模組；要啟用這條路線時，將 `whisper_train.peft.adapter_scope` 改成 `language`，並以 `active_language` 指定當次訓練語言：

- `zh-TW -> zh_tw`
- `nan-tw -> nan_tw`

設定片段如下：

```yaml
adapter_scope: language
active_language: nan-tw
```

`adapter_scope: shared` 則作為單一共享低秩模組基線。

## 信心閥值路由

`scripts/route_confidence.py` 提供可測的路由函式。路由輸入是語言機率或語言 logits，輸出是 adapter 名稱、權重與路由模式：

- 高信心：啟用最高機率語言的 adapter。
- 中信心：啟用前幾個候選 adapter 並加權。
- 低信心：啟用共享 adapter，降低錯誤硬選風險。

低信心只改變 adapter 路由，不改變任務目標；`nan-tw` 樣本仍應輸出台語文字。

`scripts/lora_adapters.py` 的 `activate_routed_adapters()` 可把路由結果套到低秩適應模型：單一或共享模式會直接切換 adapter；混合模式會在支援的模型上建立加權 adapter。

## 建議實驗順序

1. 執行 `scripts/prepare_cv.py` 產生整併 TSV。
2. 執行 `scripts/train_baseline.py` 訓練不使用低秩適應的 Whisper-medium 基線。
3. 執行 `scripts/train_lora.py` 訓練共享 `adalora` 自適應低秩主方法。
4. 將 `configs/config.yaml` 的 `method` 改為 `lora` 後訓練固定秩對照。
5. 將 `adapter_scope` 改為 `language`，分別設定 `active_language: zh-TW` 與 `active_language: nan-tw` 訓練語言專屬 adapter。
6. 使用 `scripts/route_confidence.py` 檢查閥值策略，再接入推論流程。
