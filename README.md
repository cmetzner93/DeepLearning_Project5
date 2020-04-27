# DeepLearning_Project5

## Added scripts
- cDCGAN\_utils.py added containing functions to save images and write text files

## Data: 
- MIAS MiniMammographic Database (i.e. mini-MIAS database of mammograms)
- Image size: 1024 x 1024 pixels
- Link: https://www.mammoimage.org/databases/

## Using GPU via GCP
- Issues using python script with cloned github repo
- images in github repo were not properly copied
## Different approach
- Using google cloud storage - google bucket
- Uploaded scripts and images to own google bucket related to project in GCP
- Transferred data on bucket to virtual machine (instance) via following code:
- gsutil -m cp -r gs://dl\_final\_proj/* <designated directory>

- After fixing problems with recognizing path (cv2.imread(image_path))
- Problems occured with the memory (RAM) on the used machine
- Increased number of virtual cores by 8 to 30GB RAM

- ran script for 200 epochs, batch_size: 20 for real and fake images.
- start seeing some form of breast / mammogramm
- more training cycles required to achieve better images


## Goal with real classifier:
- Need classifier (anna)
- Classify images into classes (normal, benign, malignant) w.r.t origin (real/fake)


## Modification of cDCGAN\_train.py script
- Add model save feature
- Plots (accuracy, loss vs epoch)
- frechet-distance score - Kullback-Leibler Divergence
