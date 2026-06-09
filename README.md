# Whisper-TW

專案目標是做自動語音辨識：

- `zh-TW` 音訊辨識成中文文字
- `nan-tw` 音訊辨識成台語文字

訓練用 YAML 只保留 `configs/config.yaml` 與 `configs/config_h100.yaml`。語言差異由 `--language` 指定，不再使用多份語言設定檔。

## 1. 建立環境

若 Linux 主機沒有 conda，先安裝 Miniforge：

- Windows：不適用
- Linux：`wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh`
- Linux：`bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"`
- Linux：`source "$HOME/miniforge3/etc/profile.d/conda.sh"`
- Linux：`conda init bash`
- Linux：`source ~/.bashrc`

建立乾淨 conda 環境：

- Windows：`conda create -n whisper-tw python=3.11 -y`
- Linux：`conda create -n whisper-tw python=3.11 -y`

啟用環境：

- Windows：`conda activate whisper-tw`
- Linux：`conda activate whisper-tw`

安裝 PyTorch 與 CUDA 套件：

- Windows：`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y`
- Linux：`pip uninstall -y torch torchaudio torchvision`
- Linux：`pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118`

安裝音訊相依套件：

- Windows：`conda install -c conda-forge ffmpeg libsndfile -y`
- Linux：`conda install -c conda-forge ffmpeg libsndfile -y`

安裝專案 Python 套件：

- Windows：`pip install -r requirements.txt`
- Linux：`pip install -r requirements.txt`

確認 CUDA 可用：

- Windows：`python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"`
- Linux：`python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"`

確認 Linux 主機可看到 H100：

- Windows：不適用
- Linux：`nvidia-smi`

## 2. 下載與前處理資料

訓練資料來自 Mozilla Data Collective 的 Common Voice，下載需要 API 金鑰。
下載所有已知 Common Voice 資料集。若資料夾已存在，腳本會自動略過並繼續下一份：

- Windows：`python .\scripts\download_cv.py --output-dir .\data`
- Linux：`python scripts/download_cv.py --output-dir data`

整理 `zh-TW` 與 `nan-tw` 欄位，產生雙語訓練 TSV。`nan-tw` 的括號台羅或白話字會移到輔助欄位，主目標保留台語文字：

- Windows：`python .\scripts\prepare_cv.py --data-root .\data --output-dir .\data\processed\common_voice`
- Linux：`python scripts/prepare_cv.py --data-root data --output-dir data/processed/common_voice`

若在記錄檔環境不想顯示進度條：

- Windows：`python .\scripts\prepare_cv.py --data-root .\data --output-dir .\data\processed\common_voice --no-progress`
- Linux：`python scripts/prepare_cv.py --data-root data --output-dir data/processed/common_voice --no-progress`

檢查前處理後的欄位與音檔路徑：

- Windows：`python .\scripts\check_cv.py --prepared-dir .\data\processed\common_voice --data-root .\data --check-audio --strict`
- Linux：`python scripts/check_cv.py --prepared-dir data/processed/common_voice --data-root data --check-audio --strict`

## 3. wandb 記錄

訓練預設會把訓練損失、學習率與驗證損失紀錄到 wandb。第一次訓練前請先登入：

- Windows：`wandb login`
- Linux：`wandb login`

若要避免互動式登入，也可以用環境變數：

- Windows：`$env:WANDB_API_KEY="你的 wandb 金鑰"`
- Linux：`export WANDB_API_KEY="你的 wandb 金鑰"`

## 4. 訓練

訓練對比式查詢路由器。路由器會使用凍結的語音編碼器表示產生查詢向量，並和 `zh-TW`、`nan-tw` 語言原型向量比對。若論文實驗同時比較 `openai/whisper-small`、`openai/whisper-medium` 與 `openai/whisper-large-v3-turbo`，三個底座模型需各自訓練一個路由器：

- Windows 小型：`python .\scripts\train_contrastive_router.py --config .\configs\config_h100.yaml --model-name-or-path openai/whisper-small --output-dir artifacts\models\contrastive_router_whisper_small_h100 --feature-cache-dir artifacts\models\contrastive_router_whisper_small_h100\feature_cache --dataloader-num-workers 0`
- Windows 中型：`python .\scripts\train_contrastive_router.py --config .\configs\config_h100.yaml --model-name-or-path openai/whisper-medium --output-dir artifacts\models\contrastive_router_whisper_medium_h100 --feature-cache-dir artifacts\models\contrastive_router_whisper_medium_h100\feature_cache --dataloader-num-workers 0`
- Windows 大型高速版：`python .\scripts\train_contrastive_router.py --config .\configs\config_h100.yaml --model-name-or-path openai/whisper-large-v3-turbo --output-dir artifacts\models\contrastive_router_whisper_large_v3_turbo_h100 --feature-cache-dir artifacts\models\contrastive_router_whisper_large_v3_turbo_h100\feature_cache --dataloader-num-workers 0`
- Linux 小型：`python scripts/train_contrastive_router.py --config configs/config_h100.yaml --model-name-or-path openai/whisper-small --output-dir artifacts/models/contrastive_router_whisper_small_h100 --feature-cache-dir artifacts/models/contrastive_router_whisper_small_h100/feature_cache --dataloader-num-workers 0`
- Linux 中型：`python scripts/train_contrastive_router.py --config configs/config_h100.yaml --model-name-or-path openai/whisper-medium --output-dir artifacts/models/contrastive_router_whisper_medium_h100 --feature-cache-dir artifacts/models/contrastive_router_whisper_medium_h100/feature_cache --dataloader-num-workers 0`
- Linux 大型高速版：`python scripts/train_contrastive_router.py --config configs/config_h100.yaml --model-name-or-path openai/whisper-large-v3-turbo --output-dir artifacts/models/contrastive_router_whisper_large_v3_turbo_h100 --feature-cache-dir artifacts/models/contrastive_router_whisper_large_v3_turbo_h100/feature_cache --dataloader-num-workers 0`

也可以用專用批次腳本依 `configs/baselines.yaml` 的 `router_jobs` 自動補齊三個路由器；若某個型號已有 `contrastive_router.pt`，預設會略過：

- Windows：`python .\scripts\run_routers.py --config .\configs\config_h100.yaml --dataloader-num-workers 0`
- Linux：`python scripts/run_routers.py --config configs/config_h100.yaml --dataloader-num-workers 0`

若要強制重跑全部路由器：

- Windows：`python .\scripts\run_routers.py --config .\configs\config_h100.yaml --redo --dataloader-num-workers 0`
- Linux：`python scripts/run_routers.py --config configs/config_h100.yaml --redo --dataloader-num-workers 0`

訓練 `zh-TW` 語言專屬 AdaLoRA adapter：

- Windows：`python .\scripts\train.py --config .\configs\config.yaml --language zh-TW`
- Linux：`python scripts/train.py --config configs/config_h100.yaml --language zh-TW`

訓練 `nan-tw` 語言專屬 AdaLoRA adapter：

- Windows：`python .\scripts\train.py --config .\configs\config.yaml --language nan-tw`
- Linux：`python scripts/train.py --config configs/config_h100.yaml --language nan-tw`

單獨微調一個 Whisper 基線模型：

- Windows：`python .\scripts\ft_whisper.py --config .\configs\config.yaml --language zh-TW --model-name-or-path openai/whisper-small`
- Linux：`python scripts/ft_whisper.py --config configs/config_h100.yaml --language zh-TW --model-name-or-path openai/whisper-small`

微調全部 Whisper 基線模型：

- Windows：`python .\scripts\run_baselines.py --config .\configs\config.yaml --redo-ft`
- Linux：`python scripts/run_baselines.py --config configs/config_h100.yaml --redo-ft`

訓練全部本研究自適應低秩轉接模組。此腳本會依 `configs/baselines.yaml` 的 `adalora_jobs` 依序訓練三個型號與兩種語言；若某個型號與語言已有權重，預設會略過：

- Windows：`python .\scripts\run_adalora.py --config .\configs\config_h100.yaml --dataloader-num-workers 0`
- Linux：`python scripts/run_adalora.py --config configs/config_h100.yaml --dataloader-num-workers 0`

若要強制重跑全部本研究自適應低秩轉接模組：

- Windows：`python .\scripts\run_adalora.py --config .\configs\config_h100.yaml --redo --dataloader-num-workers 0`
- Linux：`python scripts/run_adalora.py --config configs/config_h100.yaml --redo --dataloader-num-workers 0`

## 5. 評估

若目前只完成其中一個語言 adapter，可以先單獨評估該語言。以下指令只會讀取對應 `language_label` 的測試樣本，並載入對應 adapter。

評估 `zh-TW` adapter：

- Windows：`python .\scripts\evaluate.py --config .\configs\config.yaml --mode single --language zh-TW --split test`
- Linux：`python scripts/evaluate.py --config configs/config_h100.yaml --mode single --language zh-TW --split test`

評估 `nan-tw` adapter：

- Windows：`python .\scripts\evaluate.py --config .\configs\config.yaml --mode single --language nan-tw --split test`
- Linux：`python scripts/evaluate.py --config configs/config_h100.yaml --mode single --language nan-tw --split test`

等 `zh-TW`、`nan-tw` adapter 與對比式路由都完成後，可以執行完整路由評估。這會同時計算路由指標、路由選擇 adapter 的 CER、正確 adapter 的 CER 與錯誤 adapter 的 CER：

- Windows：`python .\scripts\evaluate.py --config .\configs\config.yaml --mode router --split test`
- Linux：`python scripts/evaluate.py --config configs/config_h100.yaml --mode router --split test`

若只要重跑訓練完成的對比式路由器單獨驗證，不載入任何轉接模組，需指定對應底座模型與同型號路由器權重：

- Windows 小型：`python .\scripts\evaluate.py --config .\configs\config_h100.yaml --mode router_metrics --model-name-or-path openai/whisper-small --router-checkpoint artifacts\models\contrastive_router_whisper_small_h100\contrastive_router.pt --split test`
- Windows 中型：`python .\scripts\evaluate.py --config .\configs\config_h100.yaml --mode router_metrics --model-name-or-path openai/whisper-medium --router-checkpoint artifacts\models\contrastive_router_whisper_medium_h100\contrastive_router.pt --split test`
- Windows 大型高速版：`python .\scripts\evaluate.py --config .\configs\config_h100.yaml --mode router_metrics --model-name-or-path openai/whisper-large-v3-turbo --router-checkpoint artifacts\models\contrastive_router_whisper_large_v3_turbo_h100\contrastive_router.pt --split test`
- Linux 小型：`python scripts/evaluate.py --config configs/config_h100.yaml --mode router_metrics --model-name-or-path openai/whisper-small --router-checkpoint artifacts/models/contrastive_router_whisper_small_h100/contrastive_router.pt --split test`
- Linux 中型：`python scripts/evaluate.py --config configs/config_h100.yaml --mode router_metrics --model-name-or-path openai/whisper-medium --router-checkpoint artifacts/models/contrastive_router_whisper_medium_h100/contrastive_router.pt --split test`
- Linux 大型高速版：`python scripts/evaluate.py --config configs/config_h100.yaml --mode router_metrics --model-name-or-path openai/whisper-large-v3-turbo --router-checkpoint artifacts/models/contrastive_router_whisper_large_v3_turbo_h100/contrastive_router.pt --split test`

若要同時比較目前轉接模組與 `configs/baselines.yaml` 中啟用的基線：

- Windows：`python .\scripts\run_baselines.py --config .\configs\config_h100.yaml --split test --dataloader-num-workers 0`
- Linux：`python scripts/run_baselines.py --config configs/config_h100.yaml --split test --dataloader-num-workers 0`

若要自動補齊缺少的一般微調、自適應低秩與路由器權重，並接著輸出三張論文比較表：

- Windows：`python .\scripts\run_baselines.py --config .\configs\config_h100.yaml --split test --dataloader-num-workers 0`
- Linux：`python scripts/run_baselines.py --config configs/config_h100.yaml --split test --dataloader-num-workers 0`

## 6. 檢查

整體流程靜態總檢查：

- Windows：`python .\scripts\check_pipeline.py`
- Linux：`python scripts/check_pipeline.py`

真實資料下載並完成前處理後，可要求總檢查同時驗證實際 TSV 與音檔路徑：

- Windows：`python .\scripts\check_pipeline.py --require-real-data --data-root .\data --prepared-dir .\data\processed\common_voice`
- Linux：`python scripts/check_pipeline.py --require-real-data --data-root data --prepared-dir data/processed/common_voice`
