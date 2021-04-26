# Person_Segmentation

# Train
* python train.py --train_path data\coco2_dataset\images --workdir data --model_type mobilenetV2 --batch_size 8 --epoch 15

# Test
## GPU
* python predict.py -p data\Test_images --model_path data\mobilenetV2_model --gpu -1
## CPU
* python predict.py -p data\Test_images --model_path data\mobilenetV2_model

# Summary
1.Network Architecture
* Based on the model size and inference time requirements ,Small UNet based
  architecture with MobileNetV2 Backbone is used as downsample head with
  standard transpose convolutions  along with Inverted Residuals for upsampling. 
  Used corresponding skip connections from MobileNetV2 to concatanate feature maps
  at the decoder section.

* Since Architecture uses Depth wise separable convolutions through out network ,
  no: of computations needed and total learnable parameters are greatly reduced.
* 
  

2.Training 
* Extracted COCO dataset with 65k images
* Due to Lack of GPU resources , trained model only on ~5k images
* Used Binary cross entropy and Dice as Loss functions
* Trained around 15 epochs on local machine for few hours - Laptop NVIDIA GEFORCE GTX Ti
* Recorded and plotted training results in graphs (acc,val ,loss ,metrics,epochs etc)

3.Future work
* Train more epochs to boost accuracy and reduce loss further
* Train model with Entire COCO 65k images
* Also if possible Train model with Large datasets 
   Eg:combination of all datasets COCO ,VOC,Supervisely,LIP

* Investigate model performance with other Edge AI models like ESPNet ,ENet,SqueezeNet Etc.
* Perform Quanization Aware training with Int8 precision and Benchmark performance
* Compress model with pruning and Quanization.
* Deploying model on Edge devices to verify the real time Inference speeds and tune model for further improvements.
