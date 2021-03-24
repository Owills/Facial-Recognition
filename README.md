# Facial-Recognition

.gitattributes
  Some files are very large, use git-lfs to upload datasets or trained models.
  
 .ckpt files are training models (Oliver are custem made newworks, resnet18 uses prebuilt resnet architecture)
 
 buildMaskDetection.py
  builds mask detection modes using either custum or resnet architecture. Use mask_nomask_filtered just to test and make sure everything is ready. 
  Use mask_nomask_large or find your own when you are ready to train a model)
  
testMaskDetection.py
  Prints accuracies of mask detections models on training and validation sets.
  
 maskDetectionWebcam(noFaceDetection).py
  Uses webcam to test the mask detection. **use centercrop transformation for OliverMaskDetection.ckpt
  
facenet.py
  Uses prebuilt pretrained model to locate faces on webcam and draw rectangles around each face.
  
maskDetectionWecpcam.py
  using facenet model to take cropped image around a face and print if it has a mask or not. **may not be able to detect faces, and therefore run when wearing certain masks.
  
VideoToImage.py
  take .mp4 video and coverts each frame into an image. Use this to build dataset for facial recognition.
  
covertToFolderFormat.py
  
  
BuildFaceDetection.py

to be finished later...
  
