lr global cosine LR over 0 → 60000 2e-4~1e-5 我分三次跑1-10000 10000-35000 35000-60000
用HAT_SRx4.pth 
clean threshold = 19.0
batch size = 2
EMA_DECAY = 0.9995
8x TTA
其他基本跟競賽的code裡面一個PSNR=33.5的做法差不多

	      10000  35000  60000
private score 32.978 33.423 33.607
public score  32.984 33.401 33.613

整體說明:

HAT 官方設定中，SRx4 full HAT 的基本架構是
| 參數              | 使用值           | 說明                                     
| ----------------- | --------------: | -------------------------------------- |
| `type`            |           `HAT` | Hybrid Attention Transformer           
| `upscale`         |               4 | x4 SR                                  
| `in_chans`        |               3 | RGB input                              
| `img_size`        |              64 | base config 中的 LR patch reference size 
| `window_size`     |              16 | window attention 大小                    
| `embed_dim`       |             180 | feature channel 數                      
| `depths`          | `[6,6,6,6,6,6]` | 6 個 stage，每個 stage 6 blocks            
| `num_heads`       | `[6,6,6,6,6,6]` | 每個 stage 6 heads                      
| `mlp_ratio`       |               2 | MLP hidden dim = 2 × embed dim         
| `compress_ratio`  |               3 | channel attention 壓縮比例                
| `squeeze_factor`  |              30 | channel squeeze 參數                     
| `conv_scale`      |            0.01 | convolution branch scaling             
| `overlap_ratio`   |             0.5 | overlapping cross-attention 比例         
| `upsampler`       |  `pixelshuffle` | 上採樣方式                                  
| `resi_connection` |         `1conv` | residual connection 使用 1 convolution   
| `img_range`       |             1.0 | input range 為 0~1                      


兩個模型最重要差異是這幾個：
| 參數             | `HAT_SRx4.pth` full HAT | `HAT-S_SRx4.pth` HAT-S 
| ---------------- | ----------------------: | ---------------------: 
| `embed_dim`      |                     180 |                144 
| `compress_ratio` |                       3 |                 24
| `squeeze_factor` |                      30 |                 24
| Params           |                 約 20.8M |             約 9.6M 
| Multi-Adds       |                約 102.4G |             約 54.9G 



DataLoader / Augmentation 參數
| 參數                    | 使用值 
| ----------------------- | -------------------: 
| Dataset type            | `PairedImageDataset` 
| GT patch size           |      `GT_SIZE = 256` 
| LR patch size           | 64 × 64，因為 scale = 4 
| Batch size              |     `BATCH_SIZE = 2` 
| Shuffle                 |                 True 
| Workers per GPU         |                    2 
| Dataset enlarge ratio   |                    1 
| Prefetch mode           |               `cuda` 
| Pin memory              |                 True 
| Horizontal flip         |                 True 
| Rotation augmentation   |                 True 
| Validation during train |                False 

訓練參數
| 參數                   | 使用值 
| ---------------------- | --------------: 
| Total target iteration |         `60000` 
| Iterations per session |         `10000` 
| Auto resume            |            True 
| Warmup                 | `-1`，不使用 warmup 
| Loss                   |         L1 Loss 
| Loss weight            |             1.0 
| Reduction              |            mean 
| EMA decay              |        `0.9995` 
| Save checkpoint freq   |          `1000` 
| Print freq             |           `100` 
| TensorBoard logger     |           False 
| Progress bar           |            True 

Optimizer / Scheduler 參數
| 類別           |  使用值 
| -------------- | -----------------------: 
| Optimizer      |                     Adam 
| Learning rate  |                   `2e-4` 
| Betas          |            `[0.9, 0.99]` 
| Weight decay   |                    `0.0` 
| Scheduler      | CosineAnnealingRestartLR 
| Cosine period  |                `[60000]` 
| Eta min        |                   `1e-5` 
| Restart weight |                  `[1.0]` 


訓練流程

1. 環境建立
   安裝 HAT、BasicSR、timm、opencv、scikit-image 等套件，
   並 clone XPixelGroup/HAT repository。

2. 載入資料集
   從 Kaggle input 中自動尋找 train/hr、train/lr、test/lr、
   sample_submission.csv。

3. 建立 clean training set
   對每一組 LR/HR pair：
   先將 LR 用 bicubic 放大到 HR 尺寸，
   再計算 bicubic PSNR。
   只保留 PSNR ≥ 19.0 的 pair，
   並建立 clean_train/hr 和 clean_train/lr。

4. 載入 pretrained HAT SRx4 model
   使用 HAT_SRx4.pth 作為 pretrained checkpoint，
   載入 params_ema 權重，並使用 strict loading。

5. 建立訓練設定
   使用官方 HAT SRx4 fine-tuning yaml 為基礎，
   但改成 Kaggle video game SR dataset。
   訓練 patch size 為 256，batch size 為 2，
   使用 hflip 和 rotation augmentation。

6. 訓練模型
   使用 Adam optimizer，learning rate = 2e-4，
   loss function 為 L1 loss。
   使用 cosine learning rate scheduler，
   learning rate 從 2e-4 下降到 1e-5。
   總訓練目標為 60000 iterations，
   每次 session 訓練 10000 iterations，
   並自動從前一次 checkpoint resume。

7. EMA 權重更新
   訓練過程使用 EMA decay = 0.9995，
   推論時優先使用 params_ema 權重，
   讓輸出結果較穩定。

8. 儲存 checkpoint 與 resume package
   每 1000 iterations 儲存 checkpoint。
   訓練後輸出 net_g_latest.pth、net_g_{iter}.pth，
   並打包 resume_package 方便下一次繼續訓練。

9. 推論與產生 submission
   使用最新 checkpoint 進行測試集推論。
   推論前先用 reflect padding 補到 window_size=16 的倍數，
   推論後裁回原本 x4 尺寸。

10. 使用 8x TTA
    對每張測試圖做 8 種翻轉/轉置增強，
    將 8 個輸出反轉回原方向後取平均，
    最後 clamp 到 0~1 並轉成 uint8。

11. 產生 Kaggle 上傳檔
    將 RGB output 轉成 BGR，
    再使用 RLE + zlib + base64 編碼，
    輸出 submission.csv。
