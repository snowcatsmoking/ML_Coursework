# 1 Author

**Student Name**:  
**Student ID**:  

# 2 Problem formulation

This experiment addresses a music identification classification problem. The input is a short Hum/Whistle audio clip from a participant, and the output is one of 8 song labels (an 8-class classification task). The main challenges come from strong inter-person variability, uneven audio quality, and the spectral differences between hums and whistles, which cause large variation for the same song across different singers and vocalization styles. Because the dataset is perfectly balanced, we need a clear overall metric while also examining class-level separability and confusion patterns.

# 3 Methodology

We follow a data-driven workflow: first explore and visualize the data, then design features based on observations, and finally train and evaluate models. The train/validation/test split is grouped by participant to avoid data leakage (the same participant appearing in both train and test). The split uses 80% of participants for training, with a validation subset carved out from the training participants (about 64% train, 16% validation, 20% test overall).

The training task is multi-class classification on extracted audio features. Candidate features include time-domain features (e.g., power, zero crossing rate), frequency-domain features (e.g., pitch_mean, pitch_std, spectral_centroid, spectral_rolloff), and timbral features (e.g., MFCCs). We use box/violin plots and correlation analysis to keep features with high discriminative power while reducing redundancy, forming the final feature set.

Model selection follows a classic baseline approach with four standard models: SVM (RBF), SVM (Linear), Random Forest, and k-NN. Features are standardized before training; models are evaluated on the validation set to select the best one, and final results are reported on the test set. Performance is defined primarily by Accuracy, complemented by per-class Precision/Recall, Macro F1, and the confusion matrix; group-wise cross-validation is used when needed to assess stability.
