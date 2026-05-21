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
