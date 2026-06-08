# Whisper-TW 架構白話詳解

這份文件用比較白話的方式說明目前專案架構，重點放在三個部分：

1. 共享 `Whisper-medium` 底座。
2. 語言專屬 `AdaLoRA` adapter。
3. 對比式鑰匙查詢路由器。

目標是幫助理解目前實作「到底在訓練什麼」、「自適應低秩是什麼意思」，以及「對比式路由器如何選擇語言 adapter」。

## 1. 整體架構

目前架構可以先用一句話理解：

```text
Whisper-medium 共享底座
        +
zh-TW / nan-tw 語言專屬 AdaLoRA adapter
        +
對比式路由器選擇要使用哪個 adapter
```

也就是說，本專案不是為 `zh-TW` 和 `nan-tw` 各自訓練一整個 Whisper 模型，而是使用同一個 `openai/whisper-medium` 作為共同底座，再為不同語言接上不同的小型可訓練 adapter。

白話來說：

```text
Whisper-medium 是共同的大腦。
zh_tw adapter 是華語任務的小型專門模組。
nan_tw adapter 是台語任務的小型專門模組。
路由器負責判斷輸入音訊比較應該交給哪個 adapter。
```

設定檔中的核心設定如下：

```yaml
model_name_or_path: openai/whisper-medium

peft:
  enabled: true
  method: adalora
  adapter_scope: language
  language_adapters:
    zh-TW: zh_tw
    nan-tw: nan_tw
```

因此，訓練華語時會啟用 `zh_tw` adapter，訓練台語時會啟用 `nan_tw` adapter。

## 2. 語言專屬 adapter 怎麼訓練

訓練時，指令中的 `--language` 會決定這次要訓練哪一個語言 adapter：

```bash
python scripts/train.py --config configs/config_h100.yaml --language zh-TW
python scripts/train.py --config configs/config_h100.yaml --language nan-tw
```

當指定：

```bash
--language zh-TW
```

程式會做幾件事：

- 只讀取 `language_label == zh-TW` 的資料。
- 啟用 `zh_tw` adapter。
- 輸出到華語 adapter 專屬資料夾。
- 建立華語 adapter 專屬的 wandb run name。

當指定：

```bash
--language nan-tw
```

程式則會：

- 只讀取 `language_label == nan-tw` 的資料。
- 啟用 `nan_tw` adapter。
- 輸出到台語 adapter 專屬資料夾。
- 建立台語 adapter 專屬的 wandb run name。

白話說：

```text
訓練華語時，只更新華語 adapter。
訓練台語時，只更新台語 adapter。
Whisper-medium 底座不會被完整重新訓練。
```

## 3. AdaLoRA 接在哪裡

目前 AdaLoRA 不是接在整個模型的所有地方，而是接在 Whisper 中幾個關鍵模組：

```yaml
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - out_proj
  - fc1
  - fc2
```

這些模組大致可以分成兩類：

- `q_proj`、`k_proj`、`v_proj`、`out_proj`：注意力機制相關投影。
- `fc1`、`fc2`：前饋網路相關轉換。

白話說，AdaLoRA 是在 Whisper 重要的訊息轉換位置旁邊加上小型修正模組。

原本的 Whisper 可以想成：

```text
輸入特徵 -> Whisper 原本權重 W -> 輸出
```

加入 AdaLoRA 後，變成：

```text
輸入特徵 -> Whisper 原本權重 W -> 輸出
        -> AdaLoRA 小模組 ΔW -> 補充修正
```

用簡化數學式表示：

```latex
W' = W + \Delta W
```

其中：

```latex
\Delta W = BA
```

`W` 是 Whisper 原本的權重，`BA` 是 adapter 學到的低秩修正。

## 4. 什麼是低秩

一般完整權重矩陣可能很大，例如：

```latex
W \in \mathbb{R}^{d \times k}
```

如果直接訓練整個 \(W\)，參數量會是：

```latex
d \times k
```

LoRA / AdaLoRA 不直接學完整矩陣，而是學兩個比較小的矩陣：

```latex
B \in \mathbb{R}^{d \times r}
```

```latex
A \in \mathbb{R}^{r \times k}
```

它們相乘後形成修正量：

```latex
\Delta W = BA
```

這裡的 \(r\) 就是「秩」。如果 \(r\) 越大，adapter 的表達能力越強，但參數量也越多。

低秩 adapter 的參數量約為：

```latex
r(d + k)
```

舉例，如果：

```latex
d = 1024,\quad k = 1024
```

當：

```latex
r = 16
```

參數量約為：

```latex
16(1024 + 1024) = 32768
```

如果：

```latex
r = 8
```

參數量約為：

```latex
8(1024 + 1024) = 16384
```

所以 \(r\) 越大，該位置可以學的修正越多；\(r\) 越小，該位置的 adapter 越省參數。

## 5. AdaLoRA 的自適應是什麼意思

固定 LoRA 通常會讓每一個目標模組使用固定的秩，例如每個地方都用：

```text
r = 16
```

AdaLoRA 的想法是：

```text
不是每一層、每一個矩陣都一樣重要。
重要的位置多保留一些有效秩。
不重要的位置少保留一些有效秩。
```

目前設定是：

```yaml
adalora:
  init_r: 32
  target_r: 16
  tinit: 200
  tfinal: 1000
  deltaT: 10
```

白話說：

- `init_r: 32`：一開始給比較大的低秩空間，讓模型先探索。
- `target_r: 16`：訓練過程逐漸壓到目標低秩容量。
- `tinit`、`tfinal`、`deltaT`：控制什麼時候開始調整、什麼時候結束調整、多久更新一次。

AdaLoRA 的「把容量轉給更重要的位置」不是把檔案裡的參數真的搬來搬去，而是調整各層的有效低秩分配。

例如一開始可能概念上像：

```text
q_proj：r = 16
v_proj：r = 16
fc1：r = 16
```

訓練後發現某些位置比較重要，可能變成：

```text
q_proj：有效 r = 24
v_proj：有效 r = 8
fc1：有效 r = 16
```

這代表它把有限的低秩學習能力，多保留在對任務比較重要的地方。

因此，AdaLoRA 的重點不是讓你看到的總參數量一定不同，而是：

```text
在相近的參數預算下，提高參數使用效率。
```

也就是說，`zh_tw` 和 `nan_tw` 兩個 adapter 的總訓練參數量一樣是合理的。它們使用相同的架構與相同的低秩總預算，但各自學到的權重和有效秩分配可能不同。

## 6. 為什麼兩種語言參數量一樣

兩種語言 adapter 的參數量一樣，主要是因為它們使用同一套設定：

```yaml
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - out_proj
  - fc1
  - fc2

init_r: 32
target_r: 16
lora_alpha: 64
```

所以一開始建立 `zh_tw` 和 `nan_tw` adapter 時，兩者結構相同，總可訓練參數量也相同。

但這不代表兩個 adapter 學到的東西一樣。差異在於：

```text
zh_tw adapter 用 zh-TW 資料訓練。
nan_tw adapter 用 nan-tw 資料訓練。
```

因此它們會學到不同的 adapter 權重。

更精準地說：

```text
總參數量相同。
訓練資料不同。
權重內容不同。
有效秩分配可能不同。
語言適應效果不同。
```

## 7. 對比式路由器的任務

語言專屬 adapter 訓練完成後，推論時需要知道一段輸入音訊應該使用哪一個 adapter。

最直覺的方式是做語言分類器，但目前專案使用的是「對比式鑰匙查詢路由器」。

它的白話概念是：

```text
每段音訊會被轉成 query。
每個語言 adapter 有一把 key。
query 比較靠近哪一把 key，就使用哪一個 adapter。
```

也就是：

```text
輸入音訊
  -> Whisper encoder 抽特徵
  -> 路由器產生 query 向量
  -> 與 zh-TW key / nan-tw key 比相似度
  -> 選相似度最高的 key
  -> 使用對應 adapter
```

## 8. 路由器如何把音訊變成 query

路由器訓練時，Whisper encoder 是凍結的：

```text
Whisper encoder 只負責抽特徵，不更新參數。
```

每段音訊會先經過 Whisper encoder，得到時間序列特徵：

```text
[batch, time, hidden_size]
```

因為一段音訊有很多時間點，所以路由器使用 attention pooling 把整段音訊壓成一個句級向量。

白話說：

```text
模型會學哪些時間點比較重要。
重要時間點權重大一點。
不重要時間點權重小一點。
最後加權平均成一個整段音訊向量。
```

接著，這個句級向量會通過 `QueryProjection`，轉成固定維度的 query。

H100 設定中：

```yaml
embedding_size: 128
hidden_ratio: 0.25
dropout: 0.2
```

所以每段音訊最後會得到一個 128 維的 query 向量。

## 9. 語言 key 是什麼

路由器裡有可學習的 `adapter_keys`：

```text
zh-TW key
nan-tw key
```

每一把 key 都是一個 128 維向量，對應一個語言 adapter。

白話說：

```text
zh-TW key 是華語 adapter 的代表向量。
nan-tw key 是台語 adapter 的代表向量。
```

路由器訓練時會讓：

```text
華語音訊 query 靠近 zh-TW key。
台語音訊 query 靠近 nan-tw key。
```

同時也會讓：

```text
華語音訊 query 遠離 nan-tw key。
台語音訊 query 遠離 zh-TW key。
```

## 10. 相似度如何計算

query 和 key 都會做正規化，所以它們的內積可以理解成餘弦相似度：

```latex
s(q, k) = q^\top k
```

如果一段音訊是華語，理想狀態是：

```text
s(query, zh-TW key) 高
s(query, nan-tw key) 低
```

如果一段音訊是台語，理想狀態是：

```text
s(query, nan-tw key) 高
s(query, zh-TW key) 低
```

推論時，路由器會選相似度最高的 key。

## 11. 對比式損失怎麼運作

目前路由器的損失函數由兩部分組成：

```text
總損失 = 交叉熵損失 + margin 權重 × margin 對比損失
```

設定如下：

```yaml
temperature: 0.1
label_smoothing: 0.05
margin: 0.2
margin_loss_weight: 0.1
```

### 11.1 交叉熵損失

交叉熵損失讓路由器能分對語言。

對一筆華語音訊，模型要讓：

```text
zh-TW 分數 > nan-tw 分數
```

對一筆台語音訊，模型要讓：

```text
nan-tw 分數 > zh-TW 分數
```

這一部分主要負責分類正確。

### 11.2 margin 對比損失

margin loss 不只要求分對，還要求正確 key 和錯誤 key 的距離要拉開。

對一筆樣本：

```text
positive = query 與正確語言 key 的相似度
negative = query 與最像的錯誤語言 key 的相似度
```

它要求：

```latex
positive - negative \geq margin
```

目前：

```latex
margin = 0.2
```

所以它希望：

```latex
s(q, k_{\text{correct}}) - s(q, k_{\text{wrong}}) \geq 0.2
```

如果差距不夠，就會被懲罰。

例如：

```text
正確 key 相似度 = 0.70
錯誤 key 相似度 = 0.60
差距 = 0.10
```

因為 `0.10 < 0.20`，所以會被懲罰。

如果：

```text
正確 key 相似度 = 0.80
錯誤 key 相似度 = 0.55
差距 = 0.25
```

因為 `0.25 >= 0.20`，所以 margin loss 不再懲罰。

白話說：

```text
交叉熵：要求選對。
margin loss：要求選對之外，還要和錯誤選項拉開距離。
```

## 12. temperature 和 label smoothing

`temperature` 控制相似度分數被放大的程度。

目前：

```yaml
temperature: 0.1
```

程式會做：

```latex
logits = \frac{s(q, k)}{temperature}
```

如果：

```text
similarity = [0.7, 0.6]
```

temperature 是 1 時：

```text
logits = [0.7, 0.6]
```

temperature 是 0.1 時：

```text
logits = [7.0, 6.0]
```

所以 temperature 越小，模型對相似度差異越敏感。

`label_smoothing` 則是避免模型過度自信。

目前：

```yaml
label_smoothing: 0.05
```

白話說，就是不要讓模型把正確類別看成絕對 100%、錯誤類別看成絕對 0%，而是保留一點彈性，讓訓練比較穩。

## 13. 對比式路由器訓練時更新哪些參數

對比式路由器訓練時：

```text
Whisper encoder：不更新
AdaLoRA adapter：不更新
attention pooling：更新
query projection：更新
adapter keys：更新
```

所以它不是在訓練 ASR 模型本身，而是在訓練一個獨立的選擇器。

這個選擇器學會：

```text
看到一段音訊後，判斷它比較應該使用 zh_tw adapter 還是 nan_tw adapter。
```

## 14. 對比式路由器的評估指標

目前路由器會評估：

- `accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `confusion_matrix`
- `avg_positive_similarity`
- `avg_max_negative_similarity`
- `avg_similarity_gap`

白話說：

- `accuracy`：整體選對比例。
- `macro_f1`：兩種語言平均後的 F1，適合看類別是否平衡。
- `confusion_matrix`：看華語是否常被判成台語，或台語是否常被判成華語。
- `avg_positive_similarity`：query 跟正確 key 平均有多像。
- `avg_max_negative_similarity`：query 跟最像的錯誤 key 平均有多像。
- `avg_similarity_gap`：正確 key 相似度和錯誤 key 相似度的差距。

其中 `avg_similarity_gap` 很適合說明對比學習效果。它越大，代表路由器越能把正確語言和錯誤語言分開。

## 15. 推論時的完整流程

推論時，概念流程如下：

```text
輸入音訊
  -> Whisper encoder 抽取音訊表示
  -> 對比式路由器產生 query
  -> query 與 zh-TW / nan-tw key 計算相似度
  -> 選出最相近的語言 key
  -> 載入對應語言 adapter
  -> Whisper + adapter 產生辨識結果
```

如果路由器選到 `zh-TW key`：

```text
使用 zh_tw adapter
```

如果路由器選到 `nan-tw key`：

```text
使用 nan_tw adapter
```

## 16. 可放入論文的方法描述草稿

以下是一段比較接近論文方法章的寫法：

```text
本研究以 Whisper-medium 作為共享語音辨識底座，並於其注意力投影層與前饋網路層導入語言專屬 AdaLoRA 適應模組。相較於為每一語言獨立微調完整模型，本研究僅針對低秩適應參數進行更新，藉此降低訓練成本並保留預訓練模型之跨語言聲學表徵。針對 zh-TW 與 nan-tw，本研究分別建立 zh_tw 與 nan_tw adapter，訓練時依語言標籤篩選資料並僅啟用對應 adapter，使各語言能在共享底座下保有獨立的參數更新路徑。

此外，本研究採用 AdaLoRA 進行自適應低秩分配。各語言 adapter 具有相同初始低秩配置與總容量預算，但訓練過程中會根據參數重要性調整不同層與矩陣的有效秩配置，使有限的低秩容量能集中於對任務較關鍵的位置。此設計並非單純增加或減少總參數量，而是在固定參數預算下提升低秩參數的使用效率。

為了在推論階段自動選擇合適的語言 adapter，本研究設計一個對比式鑰匙查詢路由器。該路由器以凍結的 Whisper encoder 作為語音表徵抽取器，先透過注意力池化將時間序列語音表徵壓縮為句級向量，再經由非線性投影層產生正規化查詢向量。每個語言 adapter 對應一個可學習鑰匙向量，路由器透過查詢向量與各語言鑰匙之間的餘弦相似度決定 adapter 選擇。訓練目標結合交叉熵損失與 margin-based 對比損失，使查詢向量不僅能分類至正確語言，也能與最接近的錯誤語言鑰匙保持最小相似度間隔。
```

## 17. 最精簡總結

最白話的總結是：

```text
本專案用同一個 Whisper-medium 當共同底座。
華語和台語各自有一個 AdaLoRA 小模組。
AdaLoRA 不是讓總參數量一定不同，而是讓有限的低秩容量用在更重要的位置。
對比式路由器則負責把音訊轉成 query，判斷它比較接近哪個語言 adapter 的 key，最後選擇對應 adapter 做辨識。
```
