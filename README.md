# Data Preparation and Benchmark Code Execution

## I. Data Preparation

### 1. Download the Original Dataset
It is recommended to use Baidu Netdisk for downloading within the country. After downloading, place it in the same directory level as `eye-tracking`.
- Baidu Netdisk link: [https://pan.baidu.com/s/1luJ84eXsaWNaed5FmKxXhA?pwd=evye](https://pan.baidu.com/s/1luJ84eXsaWNaed5FmKxXhA?pwd=evye) (Password: evye)
- Onedrive link: [https://1drv.ms/f/s](https://1drv.ms/f/s) (Please note that the link seems to be incomplete. Please check the validity of the link and try again if necessary.)
- Additionally, the GitHub repository for EV-Eye is available at: [https://github.com/Ningreka/EV-Eye?tab=readme-ov-file](https://github.com/Ningreka/EV-Eye?tab=readme-ov-file)

### 2. Prepare the Dataset
1. Locate the directory `.../EV_Eye_dataset/raw_data/Data_davis_labelled_with_mask`, which contains over 9,000 images with segmentation labels saved as `.h5` files.
```
─Data_davis_labelled_with_mask
├─left
│  ├─user1_session_1_0_2.h5
│  ├─user1_session_2_0_1.h5
│  │─user1_session_2_0_2.h5
│  ..........
├─right
│  ├─user1_session_1_0_2.h5
│  ├─user1_session_2_0_1.h5
│  │─user1_session_2_0_2.h5
│  ..........
```
2. Run the script `.../eye-tracking/EvEye/utils/scripts/h5_to_png.py` to parse the `.h5` files into `.png` images and their corresponding labels.
3. Use the script `.../eye-tracking/tools/train.py` with the configuration file `/mnt/hdd_2T_1/junyuan/eye-tracking/configs/DavisEyeEllipse_RGBUNet.yaml` to train a U-Net segmentation model.
4. Apply the trained U-Net model to the remaining data in `.../EV_Eye_dataset/raw_data/Data_davis` to obtain segmentation labels for all RGB images.
5. Run the script `.../eye-tracking/EvEye/utils/scripts/getEllipse.ipynb` to convert the segmentation labels into ellipse labels represented by five-parameter tuples. Both events and ellipse labels should be in `.txt` format.
- The format for events should be `(t,x,y,p)`:
  ```
  1726650454765424 152 64 0
  1726650454765433 160 48 0
  1726650454765443 344 64 1
  1726650454765462 254 81 0
  1726650454765464 171 128 0
  ```
- The format for ellipse labels should be `(t,x,a,b,theta)`:
  ```
  1726650454770344 189.09 141.27 40.78 44.11 127.53
  1726650454815374 189.66 140.94 41.07 44.22 115.67
  1726650454860405 189.08 141.04 40.43 43.99 134.25
  1726650454905435 189.10 141.09 40.38 43.99 135.05
  1726650454950465 189.19 141.05 40.50 44.58 125.54
  ```
6. Divide the dataset into training, validation, and test sets as needed. The organized dataset format should be:
```
─dataset
├─train
│  ├─data
│    ├─1.txt
│    ├─2.txt
│  │─ellipse
│    ├─1.txt
│    ├─2.txt
│      ..........
├─val
│  ├─data
│    ├─1.txt
│    ├─2.txt
│  │─ellipse
│    ├─1.txt
│    ├─2.txt
│      ..........
```
### 3. Memmap Caching
If you need to train a large amount of data at once, you can cache it using Memmap. The corresponding Memmap dataset code is provided for reference.
- Create the cached dataset using the script: `.../eye-tracking/EvEye/utils/cache/MemmapCacheStructedEvents.ipynb`

## II. Running Benchmark Code

The training, validation, and prediction codes are all located in `.../eye-tracking/tools`. The code is based on the Lightning library and operates by configuring different config files.

### 1. Train
```bash
python .../eye-tracking/tools/train.py --config .../eye-tracking/configs/DavisEyeEllipse_EPNet.yaml
```

### 2. Validate
```bash
python .../eye-tracking/tools/validate.py --config .../eye-tracking/configs/DavisEyeEllipse_EPNet.yaml
```
