# Whisper-TW

目前專案目標是做自動語音辨識：

- `zh-TW` 音訊辨識成中文文字
- `nan-tw` 音訊辨識成台語文字

若後續整合雙語資料或語言切換機制，方向是同一模型內以對比式鑰匙查詢路由選擇對應語言 adapter；不是把台語音訊轉成華語文字。

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

訓練預設會把訓練損失、學習率與驗證損失紀錄到 wandb。第一次訓練前請先登入：

```powershell
wandb login
```

訓練對比式鑰匙查詢路由。路由器會使用凍結的 Whisper encoder 表示產生查詢向量，並和 `zh-TW`、`nan-tw` adapter 的鑰匙向量比對：

```powershell
python .\scripts\train_contrastive_router.py --config .\configs\config.yaml
```

路由訓練結束後會輸出 test 指標與混淆矩陣圖。

訓練語言專屬 AdaLoRA adapter。這個流程對齊 LoRA-Whisper 的多語言擴充方式，但每個語言 adapter 使用 AdaLoRA 動態分配容量：

```powershell
python .\scripts\train.py --config .\configs\config.yaml --language zh-TW
python .\scripts\train.py --config .\configs\config.yaml --language nan-tw
```

訓練用 YAML 只保留 `configs/config.yaml` 與 `configs/config_h100.yaml`。語言差異由 `--language` 指定，不再使用多份語言設定檔。

若目前只完成其中一個語言 adapter，可以先單獨評估該語言。以下指令只會讀取 `language_label == zh-TW` 的測試樣本，並載入 `zh_tw` adapter：

```powershell
python .\scripts\evaluate.py --config .\configs\config.yaml --mode single --language zh-TW --split test
```

快速檢查可先限制樣本數：

```powershell
python .\scripts\evaluate.py --config .\configs\config.yaml --mode single --language zh-TW --split test --max-samples 20
```

等 `zh-TW`、`nan-tw` adapter 與對比式路由都完成後，可以執行完整路由評估。這會同時計算路由指標、路由選擇 adapter 的 CER、正確 adapter 的 CER 與錯誤 adapter 的 CER：

```powershell
python .\scripts\evaluate.py --config .\configs\config.yaml --mode router --split test
```

若只要重跑訓練完成的對比式路由測試，不載入任何 LoRA adapter：

```powershell
python .\scripts\evaluate.py --config .\configs\config.yaml --mode router_metrics --split test
```

若要同時比較目前 adapter 與 `configs/baselines.yaml` 中啟用的 Whisper 基線：

```powershell
python .\scripts\evaluate_baselines.py --config .\configs\config.yaml --split test
```

H100 / Linux 指令：

```bash
python scripts/train_contrastive_router.py --config configs/config_h100.yaml
python scripts/train.py --config configs/config_h100.yaml --language zh-TW
python scripts/train.py --config configs/config_h100.yaml --language nan-tw
```

整體流程靜態總檢查：

```powershell
python .\scripts\check_pipeline.py
```

真實資料下載並完成前處理後，可要求總檢查同時驗證實際 TSV 與音檔路徑：

```powershell
python .\scripts\check_pipeline.py --require-real-data --data-root .\data --prepared-dir .\data\processed\common_voice
```
