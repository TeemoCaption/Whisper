# Whisper-medium 雙語低秩適應實作檢核清單

## 一、資料集前處理

- 已實作 `scripts/prepare_cv.py`，整理 `zh-TW` 與 `nan-tw` Common Voice 欄位。
- `scripts/prepare_cv.py` 預設使用 tqdm 顯示語言、切分、樣本與輸出進度。
- `scripts/download_cv.py` 已支援環境變數、命令列與本機金鑰檔讀取 Mozilla Data Collective API 金鑰。
- 已依 Mozilla Data Collective `nan-tw` 頁面建議，將括號中的台羅或白話字移到 `romanization_text`，主訓練目標保留台語漢字。
- 已保留 `raw_sentence`、`target_text`、`language_label`、`target_script`、`filter_reason`，方便檢查文字標註。
- 已實作 `scripts/check_cv.py`，可檢查欄位、台語括號標音是否已移除、音檔路徑是否存在。
- 實際資料下載後需執行 `check_cv.py --check-audio --strict`，確認真實 `zh-TW` 與 `nan-tw` 欄位內容。

## 二、兩篇論文方法改良

- 已固定底座為 `openai/whisper-medium`。
- 已在 `scripts/lora_adapters.py` 支援固定秩低秩適應與自適應低秩適應。
- `configs/config.yaml` 預設使用 `adalora`，對應自適應低秩主方法。
- `configs/config.yaml` 是 8GB VRAM 訓練配置；`configs/config_h100.yaml` 是 H100 訓練配置，兩者模型、資料、低秩方法與路由設定一致，只差資源相關訓練欄位。
- `configs/config.yaml`、`configs/config_h100.yaml` 與 `configs/baseline.yaml` 都明確設定 `disable_tqdm: false`，保留訓練與評估進度條。
- 已保留 `lora` 設定，可切換為固定秩低秩對照組。
- 已定義 `zh-TW -> zh_tw` 與 `nan-tw -> nan_tw` 語言專屬轉接模組，對應 LoRA-Whisper 的多語言擴充方向。

## 三、信心閥值設計

- 已實作 `scripts/route_confidence.py`，支援高信心單一轉接模組、中信心加權混合、低信心共享轉接模組。
- 已在 `scripts/lora_adapters.py` 實作 `activate_routed_adapters()`，可將路由結果套用到低秩適應模型。
- 已確認低信心路由不會改變任務目標；`nan-tw` 仍輸出台語文字。

## 四、核心概念對齊

- 已建立 `docs/lora_confidence_method.md`，說明自適應低秩、語言專屬轉接模組與信心閥值如何對齊兩篇論文。
- 已拆分訓練入口：`scripts/train_baseline.py` 是 Whisper-medium 基線訓練，`scripts/train_lora.py` 是低秩適應訓練入口。
- 已在 `README.md` 放入資料準備、欄位檢查、基線訓練、低秩訓練、路由自測與總檢查指令。

## 五、總檢查

- 已實作 `scripts/check_pipeline.py`，用小型臨時資料檢查前處理、資料讀取、設定檔、路由與轉接模組切換。
- 已加入 `--require-real-data`，可在真實資料下載與前處理後強制檢查實際 TSV、語言標籤與音檔路徑。
- 已通過靜態整合檢查、語法檢查與信心閥值自測。
- 尚未完成真實資料層驗證，因目前工作區沒有 `data` 資料夾；下載資料後需重新執行 README 中的完整流程。
