Brain Tumor Detection Using Deep Learning
Tools & Technologies: Python, OpenCV, NumPy, Pandas, Matplotlib, Google Colab, TensorFlow/Keras (VGG19), Scikit-Learn
Project Overview:

Objective: Automatically detect presence of brain tumors in MRI scans via binary classification, achieving clinical-level sensitivity and specificity.

Data Acquisition & Preprocessing:

Curated a dataset of ~800 tumor-positive and ~800 tumor-negative T1-weighted axial MRIs.

Standardized filenames, applied contour-based cropping to isolate brain regions, resized to 240×240 and normalized pixel values.

Addressed class imbalance with targeted augmentation (rotation, zoom, brightness shifts) to boost minority-class representation.

Model Architecture & Training:

Employed a VGG19 backbone pretrained on ImageNet, replacing the top with a regularized head (Flatten → Dense(256, ReLU) → BatchNorm → Dropout → Dense(64, ReLU) → BatchNorm → Dropout → Sigmoid).

Two-stage fine-tuning:

Head-only training at learning rate 1e-3 for 6 epochs

Block-5 fine-tuning at 1e-4 for 6 epochs, with L2 regularization (1e-4), high dropout (0.6), early stopping on validation accuracy, and balanced class weights.

Results & Evaluation:

Test Accuracy: 92.3 % | Precision/Recall (N/T): 94 %/90 %, 91 %/94 %

Confusion Matrix: Balanced performance on tumor vs. non-tumor classes with high sensitivity (> 94 %).
