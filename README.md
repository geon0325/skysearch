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
We provide the codebase for training and inference of self-supervised compression models for satellite images and videos.
The deployed code can be found in the compression directory.

### 1. Image Preprocessing
Preprocess raw images and generate downscaled versions for faster ranking and embedding extraction:
```bash
python image_preprocess.py \
    --rawdatapath [RAW_IMAGE_DIR] \
    --outputpath [PREPROCESSED_IMAGE_DIR] \
    --smaller_outputdir [DOWNSCALED_IMAGE_DIR_FOR_FAST_RANKING] \
    --rate 0.50
```

### 2. Image Encoder Training and Embedding
* Generate triplet training data:
```bash
python make_traindata_image.py \
    --imagepath [PREPROCESSED_IMAGE_DIR] \
    --repeat_num 2 \
    --posdiff_list 4 \
    --negdiff_list 1 2 \
    --split_type train \
    --savename default
```

* Train the image encoder:
```bash
python train_image.py \
    --gpu 0 \
    --inputdata default \
    --channel [CHANNEL] \
    --batch_size 128 \
    --epochs 20 \
    --learning_rate 1e-6 \
    --dim 256 \
    --gamma 0.5 \
    --savedir [OUTPUT_DIR]
```
* Generate image embeddings:
```bash
python make_embeddings_image.py \
    --gpu 0 \
    --modelpath [TRAINED_IMAGE_ENCODER_PATH] \
    --outputdir [OUTPUT_DIR] \
    --outputname [OUTPUT_NAME] \
    --channel [CHANNEL]
```

### 3. Video Encoder Training and Embedding

* Prepare video sequences:
```bash
python preprocess_video.py --length 12 --gap 60
```

* Generate video triplets:
```bash
python make_traindata_video.py \
    --repeat_num 2 \
    --posdiff_list 4 \
    --negdiff_list 1 2 \
    --split_type train
```

* Train the video encoder:
```bash
python train_video.py \
    --gpu 0 \
    --inputdata default \
    --inputemb [IMAGE_EMBEDDING_PATH] \
    --channel [CHANNEL] \
    --batch_size 128 \
    --epochs 20 \
    --learning_rate 1e-6 \
    --dim 256 \
    --gamma 0.5 \
    --videolength 12 \
    --savedir [OUTPUT_DIR]
```

* Generate video embeddings
```bash
python make_embeddings_video.py \
    --gpu 0 \
    --modelpath [TRAINED_VIDEO_ENCODER_PATH] \
    --inputemb [IMAGE_EMBEDDING_PATH] \
    --outputdir [OUTPUT_DIR] \
    --outputname [OUTPUT_NAME] \
    --channel [CHANNEL]
```

---

## Video Prediction (Query Augmentation)
We leverage video prediction to augment the queries and improve long-term search quality.
The prediction code can be found [here](prediction).

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
