# Whisper-TW

目前專案目標是做自動語音辨識：

- `zh-TW` 音訊辨識成中文文字
- `nan-tw` 音訊辨識成台語文字

若後續整合雙語資料或語言切換機制，方向是同一模型內根據語言信心或語言判別結果，決定輸出中文或台語辨識結果；不是把台語音訊轉成華語文字。

建立乾淨 Conda 環境：

```powershell
conda create -n whisper-tw python=3.11 -y
conda activate whisper-tw
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c conda-forge ffmpeg libsndfile -y
pip install -r requirements.txt
```

確認 CUDA 可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

下載所有已知 Common Voice 資料集。若資料夾已存在，腳本會自動略過並繼續下一份：

```powershell
python .\scripts\download_cv.py --output-dir .\data
```

Mozilla Data Collective 下載需要 API 金鑰。建議用環境變數，或把金鑰放在本機文字檔後用 `--api-key-file` 指定：

```powershell
$env:MDC_API_KEY="你的金鑰"
python .\scripts\download_cv.py --output-dir .\data

python .\scripts\download_cv.py --output-dir .\data --api-key-file C:\tmp\mdc_api_key.txt
```

整理 `zh-TW` 與 `nan-tw` 欄位，產生雙語訓練 TSV。`nan-tw` 的括號台羅或白話字會移到輔助欄位，主目標保留台語文字：

```powershell
python .\scripts\prepare_cv.py --data-root .\data --output-dir .\data\processed\common_voice
python .\scripts\check_cv.py --prepared-dir .\data\processed\common_voice --data-root .\data --check-audio --strict
```

`prepare_cv.py` 預設會顯示 tqdm 進度條；若要在記錄檔環境關閉，可加上 `--no-progress`。

訓練預設會把損失、學習率、驗證 CER 等紀錄到 wandb。第一次訓練前請先登入：

```powershell
wandb login
```

訓練 Whisper-medium 基線模型。這個入口不啟用低秩適應，用來產生比較用基線：

```powershell
python .\scripts\train_baseline.py --config .\configs\baseline.yaml
```

訓練 Whisper-medium 低秩適應模型。這個入口用於論文方法改良，預設使用自適應低秩；若要做固定秩對照，可在 `configs/config.yaml` 的 `whisper_train.peft.method` 改成 `lora`：

```powershell
python .\scripts\train_lora.py --config .\configs\config.yaml
```

`configs/config.yaml` 是 8GB VRAM 版本；`configs/config_h100.yaml` 使用相同模型與低秩方法，只提高資源相關設定：

```bash
python scripts/train_lora.py --config configs/config_h100.yaml
```

若要訓練語言專屬轉接模組，將 `configs/config.yaml` 的 `adapter_scope` 改成 `language` 並設定 `active_language` 為 `zh-TW` 或 `nan-tw`。

信心閥值路由可先用自我檢查確認：

```powershell
python .\scripts\route_confidence.py --self-test
```

整體流程靜態總檢查：

```powershell
python .\scripts\check_pipeline.py
```

真實資料下載並完成前處理後，可要求總檢查同時驗證實際 TSV 與音檔路徑：

```powershell
python .\scripts\check_pipeline.py --require-real-data --data-root .\data --prepared-dir .\data\processed\common_voice
```
