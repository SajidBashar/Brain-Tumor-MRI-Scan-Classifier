# Brain-Tumor-MRI-Scan-Classifier

Naive machine learning driven brain tumor detection tool that classifies MRI scans as being normal or belonging to three tumor categories - glioma, pituitary, meningioma. Trained a self-defined Sequential model from Keras consisting of 5 layers on a brain tumor radiography dataset from Kaggle (https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri). The resultant convolutional neural network is summarized below: 
```
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_15 (Conv2D)           (None, 150, 150, 64)      1664      
_________________________________________________________________
max_pooling2d_15 (MaxPooling (None, 75, 75, 64)        0         
_________________________________________________________________
dropout_18 (Dropout)         (None, 75, 75, 64)        0         
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 75, 75, 128)       73856     
_________________________________________________________________
max_pooling2d_16 (MaxPooling (None, 37, 37, 128)       0         
_________________________________________________________________
dropout_19 (Dropout)         (None, 37, 37, 128)       0         
_________________________________________________________________
conv2d_17 (Conv2D)           (None, 37, 37, 128)       147584    
_________________________________________________________________
max_pooling2d_17 (MaxPooling (None, 18, 18, 128)       0         
_________________________________________________________________
dropout_20 (Dropout)         (None, 18, 18, 128)       0         
_________________________________________________________________
conv2d_18 (Conv2D)           (None, 18, 18, 128)       65664     
_________________________________________________________________
max_pooling2d_18 (MaxPooling (None, 9, 9, 128)         0         
_________________________________________________________________
dropout_21 (Dropout)         (None, 9, 9, 128)         0         
_________________________________________________________________
conv2d_19 (Conv2D)           (None, 9, 9, 256)         131328    
_________________________________________________________________
max_pooling2d_19 (MaxPooling (None, 4, 4, 256)         0         
_________________________________________________________________
dropout_22 (Dropout)         (None, 4, 4, 256)         0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 4096)              0         
_________________________________________________________________
dense_6 (Dense)              (None, 1024)              4195328   
_________________________________________________________________
dropout_23 (Dropout)         (None, 1024)              0         
_________________________________________________________________
dense_7 (Dense)              (None, 4)                 4100      
=================================================================
Total params: 4,619,524
Trainable params: 4,619,524
Non-trainable params: 0
```

After trial and error, the optimal learning rate was found with exponential decay at &alpha; = 0.0001, which seemingly provided better accuracy rates per training step. The highest accuracy of the classifier was 91.29%, shown in the following output snippet 
```
Epoch 29/30
46/46 [==============================] - 139s 3s/step - loss: 0.1450 - accuracy: 0.9516 - val_loss: 0.2515 - val_accuracy: 0.9129
```
30 epochs were ran to train the neural network with a total approximate runtime of 30 minutes for completion. Predictions are visualized as shown in the example below, 

![alt text](https://github.com/SajidBashar/Brain-Tumor-MRI-Scan-Classifier/blob/main/prediction_example.png?raw=true)

