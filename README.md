# hiding-information-based-on-NST

## Requirements

- Python 3.7
- PyTorch 1.7+
- TorchVision
- Pillow
- Skimage
- tqdm
- numpy

Anaconda environment recommended here!

(optional)

- GPU environment for training

## train

1.Secret images download [COCO](http://cocodataset.org/#download) ,
ukiyoe2photo dateset [Wikiart](https://www.kaggle.com/c/painter-by-numbers) (as style dataset and content dataset) and unzip them, rename them as `content` and `style`  respectively (recommended).

2. Modify the argument in the` train.py` such as the path of directory, epoch, learning_rate or you can add your own training code.

3. Train the model using gpu.

4. ```python
   python train.py
   
## test

1. Prepare your content image and style image. I provide some in the `content` and `style` and you can try to use them easily.

2. ```python
   python test.py
   
