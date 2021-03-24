# Facial-Recognition

**trained networks have upper nieghty percent accuracy on training and validation sets
**excuse typos

.gitattributes
  Some files are very large, use git-lfs to upload datasets or trained models.
  
 .ckpt files are training models (Oliver are custom made newworks, resnet18 uses prebuilt resnet architecture)
 
 buildMaskDetection.py
  builds mask detection models using either custom or resnet architecture. Use mask_nomask_filtered just to test and make sure everything is ready. 
  Use mask_nomask_large or find your own when you are ready to train a model)
  
testMaskDetection.py
  Prints accuracies of mask detection models on training and validation sets.
  
 maskDetectionWebcam(noFaceDetection).py
  Uses webcam to test the mask detection. **use centercrop transformation for OliverMaskDetection.ckpt
  
facenet.py
  Uses prebuilt pretrained model to locate faces on webcam and draw rectangles around each face.
  
maskDetectionWecpcam.py
  using facenet model to take cropped image around a face and print if it has a mask or not. **may not be able to detect faces, and therefore run when wearing certain masks.
  
VideoToImage.py
  take .mp4 video and coverts each frame into an image. Use this to build dataset for facial recognition.
  
covertToFolderFormat.py
  Once you have a main folder with folder inside for each face/person. Run this program on the main folder to covert it into readable training and validation set.
  
BuildFaceDetection.py
  Build face detection model using either custom or resnet architecture.

testFaceDetection.py
  Prints accuracies of face detection models on training and validation sets.
  
faceDetectionWebcam.py
  using facenet model to take cropped image around a face and prints if it is a person or not. **model is trained for my face and a small data set,, may not be accurate
  

  
