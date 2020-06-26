# Keras-Deep-CNN-for-Image-Denoising
This project mainly focuses on post-processing of images and the aim is to find the optimal solution which can maximise the image quality and calculate the difference by finding put the peek signal-to-noise ratio(PSNR). It also evaluate the performance of deep convolutional neural network used for image denosing

I include an implementation part of the image denoising model based on CNN and some previous work of DnCNN. Network Architecture of my model conatins-

• training of the dataset

• adding the gaussian noise during the backpropagation training

• use of deep convolution neural network layer where extraction of the features are done on the primary layers, processing of the images are done on the
  middle layer
  
## Results
 ### Noise Level 25
<div class="row">
  <div class="column">
    <img src="https://github.com/Emharsh/Keras-Deep-CNN-for-Image-Denoising/blob/master/results/sigma25_fig1.png" width="200" height="300"> 
    <img src="https://github.com/Emharsh/Keras-Deep-CNN-for-Image-Denoising/blob/master/results/sigma25_fig2.png" width="200" height="300">
    <img src="https://github.com/Emharsh/Keras-Deep-CNN-for-Image-Denoising/blob/master/results/sigma25_fig3.png" width="200" height="300"> 
    <img src="https://github.com/Emharsh/Keras-Deep-CNN-for-Image-Denoising/blob/master/results/sigma25_fig4.png" width="200" height="300">
  </div>
  
  ### Noise Level 25
  <div class="column">
    <img src="https://github.com/Emharsh/Keras-Deep-CNN-for-Image-Denoising/blob/master/results/sigma30_fig1.png" width="200" height="300"> 
    <img src="https://github.com/Emharsh/Keras-Deep-CNN-for-Image-Denoising/blob/master/results/sigma30_fig2.png" width="200" height="300">
    <img src="https://github.com/Emharsh/Keras-Deep-CNN-for-Image-Denoising/blob/master/results/sigma30_fig3.png" width="200" height="300"> 
    <img src="https://github.com/Emharsh/Keras-Deep-CNN-for-Image-Denoising/blob/master/results/sigma30_fig4.png" width="200" height="300">
  </div>
</div>

<h2>PSNR and SSIM</h2>

<table>
  <tr>
    <th>Level of Noise</th>
    <th>PSNR/SSIM</th>
    <th>PSNR/SSIM</th>
    <th>PSNR/SSIM</th>
    <th>PSNR/SSIM</th>
  </tr>
  <tr>
    <td >25</td>
    <td>28.39/0.84</td>
    <td>26.99/0.84</td>
    <td>26.88/0.85</td>
    <td>27.25/0.83</td>
    
  </tr>
  <tr>
    <td>25</td>
    <td>28.24/0.85</td>
    <td>26.53/0.82</td>
    <td>26.48/0.84</td>
    <td>29.88/0.877</td>
    
  </tr>

</table>

</body>
</html>


