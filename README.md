# deep-learning-challenge
Charity Funding Predictor - Deep Learning Model

Overview

This project develops a deep learning model to predict the success of charity funding applications using a dataset of nonprofit organizations. The model classifies applications as successful (1) or unsuccessful (0) based on various factors such as application type, affiliation, classification, use case, and financial information.

Project Structure

Starter_Code.ipynb - Jupyter Notebook containing data preprocessing, model training, and evaluation.

charity_model.h5 - The trained deep learning model saved in HDF5 format.

Dataset

The dataset contains the following features:

EIN, NAME (Dropped as they are non-beneficial)

APPLICATION_TYPE (Categorized and rare values replaced with "Other")

AFFILIATION, CLASSIFICATION (One-hot encoded and grouped for infrequent categories)

USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT

IS_SUCCESSFUL (Target variable: 1 = Successful, 0 = Unsuccessful)

Steps in the Notebook

Data Preprocessing:

Loads data and removes unnecessary columns.

Encodes categorical features using pd.get_dummies().

Groups rare categories under "Other".

Splits the dataset into training and testing sets.

Model Development:

Uses a Sequential Neural Network with:

Input layer: Based on feature count.

Hidden layers: 80 and 30 neurons (ReLU activation).

Output layer: 1 neuron (Sigmoid activation).

Compiles the model with binary_crossentropy loss and adam optimizer.

Model Training & Evaluation:

Trains the model for 50 epochs with batch size 32.

Evaluates accuracy using test data.

Saves the trained model as charity_model.h5.

Usage Instructions

Run the Jupyter Notebook:

Ensure all dependencies (tensorflow, pandas, sklearn) are installed.

Execute the notebook sequentially to preprocess data, train, and evaluate the model.

Load the Model for Future Predictions:

from tensorflow.keras.models import load_model
model = load_model("charity_model.h5")
predictions = model.predict(new_data)

Potential Improvements

Optimize hyperparameters (e.g., adjusting neuron count, batch size, learning rate).

Introduce dropout layers to reduce overfitting.

Experiment with more hidden layers or different activation functions.

Conclusion

This project successfully demonstrates how deep learning can be applied to predict charity funding success. With further improvements, the model can provide valuable insights to nonprofit organizations seeking funding.

