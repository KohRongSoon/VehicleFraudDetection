# VEHICLE INSURANCE FRAUD DETECTION USING MACHINE LEARNING WITH VARIOUS IMBALANCED DATASET HANDLING METHODS

Predicting vehicle fraud insurance claims is difficult as the datasets used to train the machine learning model are imbalanced. The imbalance vehicle insurance fraud dataset contains more legitimate claims compared to fraud claim data. It is difficult for machine learning models to have good performance to predict the fraud claim as the trained model with an imbalanced dataset will be biased toward the majority class. The different imbalanced datasets can be suitable for different handling techniques. Therefore, in this project, there are several proposed methods implemented to handle the imbalanced dataset before the dataset is used to train the model.

In this research project, three objectives are set which are:
1. To identify the methods for handling imbalanced datasets to improve the performance of the vehicle fraud detection model.
2. To identify the best resampling ratio of the dataset to get the optimized performance of each trained model.
3. To evaluate the performance of methods used in handling imbalanced datasets for each machine learning model.

Therefore, this research project focuses on applying several imbalanced dataset handling techniques which are **SMOTE, Random Undersampling, and Hybrid Sampling** to sample the data before the data is used for modelling. The **Ensemble Learning** of **Boosting** and **Bagging** is applied as well to handle the imbalanced dataset. The performance of the trained models by applying different techniques in handling imbalanced datasets was evaluated by using accuracy, recall, specificity and F1-Score. The best performance of the trained model which is __*Ensemble Learning of a Combination of Boosting and Decision Tree with a ratio of 0.4 undersampling*__ outperformed all the trained models.
