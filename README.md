# computer_vision
Plant Disease Detection Using Custom CNN and MobileNetV2

This repository contains the code for a computer vision project focused on plant disease detection, implemented in Google Colab (computer_vision_cw.ipynb). The project uses two models: a custom Convolutional Neural Network (CNN) and a fine-tuned MobileNetV2, to classify leaf images from peppers, potatoes, and tomatoes into 15 categories (healthy or various diseases). The MobileNetV2 model achieves higher accuracy due to transfer learning.

Project Overview





Objective: Classify plant leaf images to detect diseases or confirm healthy leaves, aiding farmers in early intervention to reduce crop losses.



Dataset: A subset of a plant disease dataset with 150 images (10 per class) across 15 classes:





Pepper: Bacterial spot, Healthy



Potato: Early blight, Late blight, Healthy



Tomato: Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites (two-spotted), Target spot, Yellow leaf curl virus, Mosaic virus, Healthy



Models:





Custom CNN: A simple CNN trained from scratch.



MobileNetV2: A pre-trained model fine-tuned for the dataset, offering superior accuracy.



Output: Models are saved as .h5 files (TensorFlow) and optionally as .pt (PyTorch for MobileNetV2). Predictions can be made on uploaded images.

Requirements





Environment: Google Colab with GPU enabled (Runtime > Change runtime type > GPU).



Python Libraries:

tensorflow==2.19.0
numpy
pandas
matplotlib
scikit-learn
seaborn
torch==2.2.0
torchvision==0.17.0
pillow



Storage: Google Drive for dataset and model storage.



Dataset: Plant disease images in /content/drive/MyDrive/computer_vision_2/dataset/, organized by class folders.

Setup Instructions





Open the Notebook:





Upload or open computer_vision_cw.ipynb in Google Colab.



Mount Google Drive:





Run the first cell to mount your Drive:

from google.colab import drive
drive.mount('/content/drive')



Install Dependencies:





Run in a Colab cell:

!pip install tensorflow==2.19.0 torch==2.2.0 torchvision==0.17.0



Dataset Preparation:





Place the dataset in /content/drive/MyDrive/computer_vision_2/dataset/ with subfolders for each class (e.g., Pepper__bell___Bacterial_spot/).



The notebook loads a subset of 150 images (10 per class). Adjust max_images_per_class to include more images if needed.

Usage

1. Dataset Loading and Preprocessing





Subset Selection: Selects 10 images per class (150 total) using os.listdir.



Splitting: Divides into training (70%, 105 images), validation (15%, 22 images), and test (15%, 23 images) sets with stratified sampling to maintain class balance.



Data Augmentation: Applies to training data (rotation, shifts, flips, rescaling) using ImageDataGenerator. Validation/test sets are only rescaled.



Generators: Creates flow_from_dataframe generators for batched loading.



Code Example:

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

2. Model Definitions





Custom CNN:





Architecture: 3 Conv2D layers (32, 64, 128 filters) with ReLU and max pooling, followed by Flatten, Dense (128, ReLU), Dropout (0.5), and Dense (15, softmax).



Compilation: Adam optimizer, categorical cross-entropy loss, accuracy metric.



Code Example:

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(15, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



MobileNetV2:





Architecture: Pre-trained MobileNetV2 (ImageNet weights) with last 20 layers unfrozen for fine-tuning. Custom head: GlobalAveragePooling2D, Dense (128, ReLU), Dropout (0.5), Dense (15, softmax).



Compilation: Adam optimizer (learning rate 0.0001), categorical cross-entropy loss, accuracy metric.



Code Example:

from tensorflow.keras.applications import MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(15, activation='softmax')(x)
transfer_model = Model(inputs=base_model.input, outputs=output)
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

3. Training





Custom CNN: Trained for 20 epochs with early stopping (patience=10 on val_loss), learning rate reduction (factor=0.2), and checkpointing to save the best model.



MobileNetV2: Trained similarly, leveraging transfer learning for faster convergence and better performance.



Code Example:

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint('/content/drive/MyDrive/custom_cnn_plant_disease.h5', monitor='val_loss', save_best_only=True)

4. Evaluation





Both models are evaluated on the test set (23 images) for accuracy, loss, precision, recall, and F1-score.



Results:





Custom CNN: ~65% accuracy, higher loss (~1.23), precision/recall/F1 ~0.64-0.65.



MobileNetV2: ~82% accuracy, lower loss (~0.57), precision/recall/F1 ~0.83.



MobileNetV2 outperforms due to pre-trained features, better handling small datasets.



Code Example:

cnn_loss, cnn_accuracy = model.evaluate(test_generator)
mobilenet_loss, mobilenet_accuracy = transfer_model.evaluate(test_generator)

5. Testing on New Images





Upload images to classify using both models.



Code Example:

from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
img, img_array = preprocess_image(image_path)
cnn_pred = model.predict(img_array)
mobilenet_pred = transfer_model.predict(img_array)



Output: Predicted class, confidence, and top-5 predictions, with the image displayed.

6. Model Comparison





MobileNetV2 consistently achieves higher accuracy and lower loss compared to the custom CNN, as shown in evaluation metrics.



Code Example:

comparison = pd.DataFrame({
    'Model': ['Custom CNN', 'MobileNetV2'],
    'Test Accuracy': [cnn_accuracy, mobilenet_accuracy],
    'Test Loss': [cnn_loss, mobilenet_loss],
    'Precision': [cnn_precision, mobilenet_precision],
    'Recall': [cnn_recall, mobilenet_recall],
    'F1-Score': [cnn_f1, mobilenet_f1]
})
print(comparison)

Model Files





Custom CNN: /content/drive/MyDrive/custom_cnn_plant_disease.h5



MobileNetV2 (TensorFlow): /content/drive/MyDrive/mobilenetv2_plant_disease.h5



MobileNetV2 (PyTorch, optional): /content/drive/MyDrive/mobilenetv2_plant_disease.pt

Notes





Dataset Size: The subset (150 images) is small for robust performance. Increase max_images_per_class (e.g., to 100) for better results.



MobileNetV2 Advantage: Higher accuracy due to transfer learning, making it ideal for small datasets.



Image Testing: Use clear RGB images of pepper, potato, or tomato leaves for accurate predictions.



PyTorch Conversion: The .pt file uses ImageNet normalization, which may slightly alter predictions compared to TensorFlow.

Troubleshooting





FileNotFoundError: Verify dataset and model paths in Google Drive.



Low Accuracy: Small dataset may cause overfitting; increase images or adjust regularization.



Upload Issues: Ensure uploaded images are valid (.jpg, .png) and <10MB.

Potential Improvements





Use a larger dataset for better generalization.



Experiment with hyperparameters (e.g., learning rate, unfrozen layers).



Add ensemble methods combining both models.



Deploy as a web or mobile app for practical use.

License

This project is for educational purposes. Ensure compliance with dataset licensing if applicable.
