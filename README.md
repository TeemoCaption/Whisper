# Whisper-TW

下載 Common Voice 資料集：

```powershell
python .\scripts\download_data.py --output-dir .\data
```

訓練 SentencePiece tokenizer：

```powershell
python .\scripts\train_tokenizer.py --config .\configs\config.yaml
```

訓練 Whisper-TW 模型：

```powershell
python .\scripts\train.py --config .\configs\config.yaml
```

評估模型：

```powershell
python .\scripts\evaluate.py --config .\configs\config.yaml --checkpoint .\artifacts\checkpoints\whisper_tw_best.pt
```

評估目前模型並與基線模型比較：

```powershell
python .\scripts\evaluate_baselines.py --config .\configs\config.yaml --checkpoint .\artifacts\checkpoints\whisper_tw_best.pt --baselines-config .\configs\baselines.yaml
```

微調 Whisper 基線模型，預設訓練 15 個 epochs，並將驗證集字錯誤率最佳的權重保存成評估可載入格式：

```powershell
python .\scripts\finetune_whisper.py --config .\configs\whisper_finetune.yaml
```
