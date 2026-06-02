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
- 已在 `scripts/lora_adapters.py` 只保留自適應低秩適應。
- `configs/config.yaml` 預設使用 `adalora`，對應自適應低秩主方法。
- `configs/config.yaml` 是 8GB VRAM 訓練配置；`configs/config_h100.yaml` 是 H100 訓練配置，兩者模型、資料與低秩方法一致，只差資源相關訓練欄位。
- `configs/config.yaml` 與 `configs/config_h100.yaml` 都明確設定 `disable_tqdm: false`，保留訓練與評估進度條。
- 已定義 `zh-TW -> zh_tw` 與 `nan-tw -> nan_tw` 語言專屬轉接模組，對應 LoRA-Whisper 的多語言擴充方向。

## 三、語言專屬 AdaLoRA 與對比式路由設計

- 低秩訓練設定只保留 `configs/config.yaml` 與 `configs/config_h100.yaml`。
- 語言專屬 AdaLoRA 不再靠多份設定檔；改由 `scripts/train.py --language zh-TW|nan-tw` 動態指定 `active_language`、`language_filter`、輸出資料夾與 wandb run name。
- 新增語言時只需新增對應語言 adapter 設定與資料篩選，不需重新訓練 Whisper 底座。
- 已新增 `whisper_tw/contrastive_router.py`，使用 Whisper encoder 表示產生查詢向量，並和語言 adapter 鑰匙向量比對。
- 已新增 `scripts/train_contrastive_router.py`，可用同一份 `config.yaml` 或 `config_h100.yaml` 訓練對比式鑰匙查詢路由，推論時用最匹配結果選擇對應 adapter。

## 四、核心概念對齊

- 已建立 `docs/language_adalora_method.md`，說明語言專屬 AdaLoRA 如何對齊兩篇論文。
- 已保留單一主訓練入口：`scripts/train.py` 是語言專屬 AdaLoRA 訓練程式。
- 已在 `README.md` 放入資料準備、欄位檢查、對比式路由訓練、語言專屬低秩訓練與總檢查指令。

## 五、總檢查

- 已實作 `scripts/check_pipeline.py`，用小型臨時資料檢查前處理、資料讀取、設定檔與低秩轉接模組設定。
- 已加入 `--require-real-data`，可在真實資料下載與前處理後強制檢查實際 TSV、語言標籤與音檔路徑。
- 已通過靜態整合檢查與語法檢查。
- 尚未完成真實資料層驗證，因目前工作區沒有 `data` 資料夾；下載資料後需重新執行 README 中的完整流程。
