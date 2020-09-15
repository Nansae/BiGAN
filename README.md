# Bidirectional GAN
The repository has implemented the **Bidirectional GAN(BiGAN)**   

**Requirements**
* python 3.6   
* tensorflow-gpu==1.14   
* pillow
* matplotlib

## Concept

## Files and Directories
* config.py : A file that stores various parameters and path settings.
* model.py : ANOGAN's network implementation file
* train.py : This file load the data and learning with BiGAN.
* utils.py : Various functions such as loading data 

## Train
1. You prepare the data.
- You can load the data by using the **read_images** function in the **utils.py**
- The structure of the data must be as follows:
   ```
   ROOT_FOLDER
      |   
      |--------SUBFOLDER (Class 0)   
      |          |------image1.jpg   
      |          |------image2.jpg   
      |          |------etc..   
      |--------SUBFOLDER (Class 1)   
      |          |------image1.jpg   
      |          |------image2.jpg   
      |          |------etc..
   ```

2. Run **train.py**
3. The result images are saved per epoch in the **sample_data**


## Reference
* Donahue, Jeff, Philipp Krähenbühl, and Trevor Darrell. "Adversarial feature learning." arXiv preprint arXiv:1605.09782 (2016).
* https://medium.com/vitalify-asia/gan-for-unsupervised-anomaly-detection-on-x-ray-images-6b9f678ca57d
* https://nbviewer.jupyter.org/github/ilguyi/gans.tensorflow.v2/blob/master/tf.v1/bigan.ipynb