# Visual Recognition using Deep Learning Project

本專案使用 **HAT-S ×4 Super Resolution model** 進行影像超解析度任務。  
主要做法是以 `HAT-S_SRx4.pth` 作為 pretrained model，搭配 clean data filtering、fine-tuning、checkpoint resume，以及 8x TTA 產生最終 submission。

---

## 1. Main Experiment Setting

| 項目 | 設定 |
|---|---|
| Model | `HAT-S_SRx4.pth` |
| Task | ×4 Super Resolution |
| TTA | 8x TTA, Test-Time Augmentation |
| Main training data | clean training dataset |
| Baseline reference | 其他基本做法與競賽中 PSNR=33.5 的 code 類似 |

Learning rate 與 iteration 設定：

| Iteration range | Learning rate setting |
|---|---|
| 1 ~ 30000 | `1e-4` constant LR |
| 30001 ~ 50000 | cosine schedule, `1e-4` → `1e-5` |
| 50001 ~ 70000 | `1e-4` constant LR, but score became worse than 50000 iter |

---

## 2. Score Result

| Iteration | 30000 | 50000 | 70000 |
|---|---:|---:|---:|
| Private score | 33.481 | **33.629** | 33.581 |
| Public score | 33.470 | **33.631** | 33.570 |

另外有使用 `iter50000` checkpoint 關掉 TTA 測試：

| Setting | Private score |
|---|---:|
| Without TTA, iter50000 | 33.594 |
| With 8x TTA, iter50000 | **33.629** |

因此本次結果顯示使用 8x TTA 會比不使用 TTA 更好。

---

## 3. Model Architecture: HAT-S SRx4

使用 **HAT-S ×4 Super Resolution model**。

| 參數 | HAT-S SRx4 使用值 | 說明 |
|---|---:|---|
| `type` | `HAT` | 一樣是 HAT 架構 |
| `upscale` | 4 | ×4 超解析度 |
| `in_chans` | 3 | RGB input |
| `img_size` | 64 | LR patch reference size |
| `window_size` | 16 | window attention 大小 |
| `embed_dim` | **144** | feature channel 數，比 full HAT 小 |
| `depths` | `[6,6,6,6,6,6]` | 6 個 stage，每個 stage 6 blocks |
| `num_heads` | `[6,6,6,6,6,6]` | 每個 stage 6 heads |
| `mlp_ratio` | 2 | MLP hidden dim = 2 × embed dim |
| `compress_ratio` | **24** | channel attention 壓縮比例 |
| `squeeze_factor` | **24** | channel squeeze 參數 |
| `conv_scale` | 0.01 | convolution branch scaling |
| `overlap_ratio` | 0.5 | overlapping cross-attention 比例 |
| `upsampler` | `pixelshuffle` | 上採樣方式 |
| `resi_connection` | `1conv` | residual connection 使用 1 convolution |
| `img_range` | 1.0 | input range 為 0~1 |
| `param_key_g` | `params_ema` | 載入 EMA 權重 |
| `strict_load_g` | true | 嚴格對齊權重名稱 |

---

## 4. Full HAT vs HAT-S Difference

兩個模型最重要差異如下：

| 參數 | `HAT_SRx4.pth` full HAT | `HAT-S_SRx4.pth` HAT-S |
|---|---:|---:|
| `embed_dim` | 180 | 144 |
| `compress_ratio` | 3 | 24 |
| `squeeze_factor` | 30 | 24 |
| Params | 約 20.8M | 約 9.6M |
| Multi-Adds | 約 102.4G | 約 54.9G |

---

## 5. Training Data Parameters

| 參數 | 值 | 意義 |
|---|---:|---|
| `BATCH_SIZE` | `4` | 每個 batch 使用 4 張 crop |
| `GT_SIZE` | `256` | HR crop 大小為 256×256 |
| `SCALE` | `4` | 放大倍率 ×4 |
| LR crop size | `64` | 由 `256 / 4` 得到 |
| `ENABLE_HOLDOUT_VAL` | `False` | 不切 validation，全部 clean data 都拿來訓練 |
| `num_worker_per_gpu` | `2` | DataLoader worker 數 |
| `dataset_enlarge_ratio` | `1` | 不額外放大 dataset |
| `prefetch_mode` | `cuda` | 使用 CUDA prefetch |
| `pin_memory` | `True` | 加速 CPU→GPU 資料搬移 |

---

## 6. Data Cleaning Parameters

| 參數 | 值 | 意義 |
|---|---:|---|
| `CLEAN_PSNR_THRESHOLD` | `18.0` | PSNR 低於 18 的 pair 視為可疑 |
| `USE_CLEAN_SYMLINK_DATA` | `True` | 建立 clean dataset 的 symlink |
| `USE_PREVIOUS_CLEAN_REPORT` | `True` | 優先使用之前算好的清理報告 |
| `PREVIOUS_CLEAN_REPORT` | `clean_pair_report.csv` | 舊的清理報告路徑 |
| `FORCE_REBUILD_CLEAN_DATA` | `False` | 不強制重算 clean report |
| `USE_PUBLIC_SUSPECT_TXT` | `True` | 如果有 suspect list，也會排除 |
| `PUBLIC_SUSPECT_PATTERNS` | `suspects*.txt` | 可疑圖片清單搜尋路徑 |

Data cleaning 流程：

```text
LR image 用 bicubic 放大到 HR 尺寸
→ 跟真正 HR image 計算 PSNR
→ 如果 PSNR < 18.0，認為 LR/HR 可能配錯
→ 丟掉該 training pair
```

建立 clean dataset：

```text
/kaggle/working/clean_train/hr
/kaggle/working/clean_train/lr
```

後續訓練使用這兩個資料夾。

---

## 7. Optimizer / Loss / LR Parameters

> 注意：LR 和實際跑的 iteration 數以最上方的 experiment setting 為準。

| 參數 | 值 | 意義 |
|---|---:|---|
| `LEARNING_RATE` | `1e-4` | Adam learning rate |
| `ADAM_BETAS` | `[0.9, 0.99]` | Adam beta 參數 |
| `WEIGHT_DECAY` | `0.0` | 不使用 weight decay |
| `EMA_DECAY` | `0.999` | EMA 權重更新係數 |
| `WARMUP_ITER` | `-1` | 不使用 warmup |
| Optimizer | `Adam` | 最佳化器 |
| Loss | `L1Loss` | 使用 L1 pixel loss |
| Scheduler | `MultiStepRestartLR` | 但實際上等於 constant LR |
| Scheduler milestone | `[10**12]` | 幾乎不會觸發 |
| Scheduler gamma | `1.0` | 即使觸發也不改變 LR |

---

## 8. Overall Training Flow

1. **準備環境**  
   安裝套件、下載 HAT repository，並載入 HAT-S SRx4 pretrained model。

2. **清理資料**  
   將 `train/lr` 用 bicubic 放大後和 `train/hr` 比較 PSNR。  
   PSNR < 18.0 的 LR/HR pair 視為可疑並移除。

3. **建立 clean training dataset**  
   使用 `clean_train/lr` 和 `clean_train/hr` 作為訓練資料。

4. **Fine-tune 模型**  
   輸入 LR crop 64×64，輸出 HR crop 256×256。  
   使用 L1 Loss 訓練 HAT-S SRx4。  
   Optimizer 使用 Adam，learning rate = `1e-4`。

5. **儲存 checkpoint**  
   每 1000 iterations 存一次模型。  
   本次主要目標訓練到 50000 iterations。

6. **推論 test/lr**  
   使用最新 checkpoint 對測試圖片做 ×4 super resolution。

7. **使用 8x TTA**  
   對圖片做翻轉、轉置推論，再把 8 個結果平均。

8. **產生 submission.csv**  
   將輸出的 HR 圖片 encode 成 Kaggle 規定格式。

---

## 9. Final Selected Model

本次實驗中，`iter50000` 的結果最好：

```text
Private score: 33.629
Public score : 33.631
```

因此最終選擇：

```text
Model checkpoint: HAT-S SRx4 iter50000
Inference method: 8x TTA
```
