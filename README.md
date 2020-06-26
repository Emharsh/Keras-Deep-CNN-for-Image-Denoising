# Keras-Deep-CNN-for-Image-Denoising
This project mainly focuses on post-processing of images and the aim is to find the optimal solution which can maximise the image quality and calculate the difference by finding put the peek signal-to-noise ratio(PSNR). It also evaluate the performance of deep convolutional neural network used for image denosing

I include an implementation part of the image denoising model based on CNN and some previous work of DnCNN. Network Architecture of my model conatins-

• training of the dataset

• adding the gaussian noise during the backpropagation training

• use of deep convolution neural network layer where extraction of the features are done on the primary layers, processing of the images are done on the
  middle layer
  
## Results
### PSNR and SSIM
<div class="row">
  <div class="column">
    <img src="https://github.com/Emharsh/Keras-Deep-CNN-for-Image-Denoising/blob/master/results/sigma25_fig1.png" width="200" height="200"> 
    <img src="https://github.com/Emharsh/Keras-Deep-CNN-for-Image-Denoising/blob/master/results/sigma25_fig2.png" width="200" height="200" padding="5px">
  </div>
</div>
