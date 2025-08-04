# SkySearch: Satellite Video Search at Scale

### Official Repository for the paper **SkySearch: Satellite Video Search at Scale** ([KDD 2025](https://kdd2025.kdd.org/) ADS Track).

![SkySearch](img/skysearch-1.png)

---

## COMS Satellite Image Dataset 
We provide **50GB** of real-world satellite images captured by the [**COMS satellite**](https://en.wikipedia.org/wiki/Chollian). The dataset is publicly available via Dropbox:

🔗 [Download COMS Dataset (Dropbox)](https://www.dropbox.com/scl/fo/xr10egc7qzvwekexs1xkx/ABHnIeb-kYh1U2PVn2-Vbhg?rlkey=nzy2zh0kak31s79twygi7gcx6&st=2mr0rd0g&dl=0)

### Folder Structure

- **Yearly folders of raw satellite images** (`2014/`, `2015/`, ..., `2019/`)
  - Each folder contains zipped image files.
  - To extract all `.zip` files at once, run:
    ```bash
    unzip '*.zip'
    ```

- **Supporting metadata files**
  - `data_list_ir02`: List of selected IR02 image paths.
  - `date_list_ir02`: Corresponding date information for each IR02 frame.

---

## Video Compression
[TODO]

---

## Video Prediction (Query Augmentation)
We leverage predictive modeling to augment video-based queries and improve search quality.

### To train the model:
```python
python train.py \
  --mse_step [MSE WEIGHT] \
  --gen_step [GENERATOR WEIGHT] \
  --disc_step [DISCRIMINATOR WEIGHT]
```

- You may additionally tune other hyperparameters such as learning rate, weight decay, etc.

### To run inference:
```python
python inference.py \
  --gpu [GPU_ID] \
  --data_path [DATA_PATH] \
  --ckpt_path [CHECKPOINT_PATH] \
  --output_path [OUTPUT_PATH]
```

- A pre-trained model checkpoint can be found in the above [Dropbox directory](https://www.dropbox.com/scl/fo/xr10egc7qzvwekexs1xkx/ABHnIeb-kYh1U2PVn2-Vbhg?rlkey=nzy2zh0kak31s79twygi7gcx6&st=2mr0rd0g&dl=0):
  - `ckpt_mse30_gen10_disc60_feat1_orth1_step80000.pth`

- The generated images (video frames) will be saved to the specified OUTPUT_PATH in the format:
```
{index}_{order}.png
- index: the test sample index
- order: the frame number (0 to 11)
```

- **Acknowledgement:** This implementation is based on the official [SimVP](https://github.com/A4Bio/SimVP) repository.
