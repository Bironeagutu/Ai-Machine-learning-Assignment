# Create an AI model that classifies plant leaf image into healthy and diseased,so smallholders farmers can identify problems earlier and reduce crop loss
Title:

Plant Disease Detection Using Transfer Learning (MobileNetV2)

1. SDG Problem Addressed

This project supports United Nations Sustainable Development Goal (SDG) 2 – Zero Hunger and SDG 12 – Responsible Consumption and Production by helping farmers identify crop diseases early.
Plant diseases can significantly reduce yields and threaten food security, especially in developing regions where expert diagnosis is limited. By using an AI-powered image classifier, farmers can quickly detect and respond to crop infections, improving productivity and reducing pesticide misuse.

2. Machine Learning Approach Used

The model uses Transfer Learning based on MobileNetV2, a lightweight Convolutional Neural Network (CNN) pre-trained on the ImageNet dataset.

Steps involved:

Data Preparation: Plant leaf images were loaded using ImageDataGenerator for real-time augmentation (rotation, flipping, zooming) to improve generalization.

Model Design: The pre-trained MobileNetV2 feature extractor was used with additional layers (Global Average Pooling, Dense, Dropout) for classification.

Training: The base model was frozen initially, then fine-tuned on the dataset with Adam optimizer and categorical cross-entropy loss.

Evaluation: The model achieved high validation accuracy in distinguishing between different plant leaf diseases and healthy leaves.

Frameworks Used:
TensorFlow & Keras (Python)

3. Results

Accuracy: The model achieved over 90% validation accuracy after fine-tuning.

Speed: Inference takes less than one second per image on a standard laptop GPU/CPU.

Output: The system predicts the plant’s health condition (e.g., “Healthy,” “Leaf Blight,” “Rust”) and provides confidence scores.

Impact: Farmers and agricultural extension officers can use a simple app or web dashboard to upload a photo and get instant results, supporting early disease control and better crop management.

4. Ethical Considerations

Data Bias: The dataset mainly includes clear images; the model may underperform on low-quality or rare diseases. Future work should include diverse field images.

Accessibility: The system should remain open-source or low-cost to ensure smallholder farmers benefit.

Environmental Impact: By detecting disease early, the model helps reduce unnecessary pesticide use, promoting eco-friendly farming.

Privacy: Only plant images are used — no human data involved.

5. Future Work

Deploy as a mobile or web application.

Add real-time camera detection for multiple leaves.

Expand dataset to cover more crop species.
