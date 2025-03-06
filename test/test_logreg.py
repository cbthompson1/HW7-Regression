"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import random

# Run with all features.
all_features = [
	'GENDER',
    'Penicillin V Potassium 250 MG',
    'Penicillin V Potassium 500 MG',
    'Computed tomography of chest and abdomen',
    'Plain chest X-ray (procedure)',
    'Diastolic Blood Pressure',
    'Body Mass Index',
    'Body Weight',
    'Body Height',
    'Systolic Blood Pressure',
    'Low Density Lipoprotein Cholesterol',
    'High Density Lipoprotein Cholesterol',
    'Triglycerides',
    'Total Cholesterol',
    'Documentation of current medications',
    'Fluticasone propionate 0.25 MG/ACTUAT / salmeterol 0.05 MG/ACTUAT [Advair]',
    '24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet',
    'Carbon Dioxide',
    'Hemoglobin A1c/Hemoglobin.total in Blood',
    'Glucose',
    'Potassium',
    'Sodium',
    'Calcium',
    'Urea Nitrogen',
    'Creatinine',
    'Chloride',
    'AGE_DIAGNOSIS',
	'NSCLC'
]

# Move most of the utility code to the file itself, outside functions.

X_train, X_val, y_train, y_val = utils.loadDataset(
	features=all_features,
	split_percent=0.8,
	split_seed=42
)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)

log_model = logreg.LogisticRegressor(
	num_feats=len(all_features)-1,
	learning_rate=0.00001,
	tol=0.01,
	max_iter=10,
	batch_size=10
)


def test_prediction():
	"""Test the bounds of the logistic regression model are between 0 and 1."""
	log_model.train_model(X_train, y_train, X_val, y_val)
	y_pred = log_model.make_prediction(X_val)
	assert max(y_pred) <= 1 and min(y_pred) >= 0

def test_loss_function():
	"""Confirm loss function works the same as sklearn's."""
	log_model.train_model(X_train, y_train, X_val, y_val)
	y_pred = log_model.make_prediction(X_val)
	pred_loss = log_model.loss_function(y_val, y_pred)
	actual_loss = log_loss(y_val, y_pred)
	assert pred_loss == actual_loss


def test_gradient():
	"""A manually calculated gradient equals the function's."""

	# Initialize weights randomly and add bias to matrix.
	log_model.W = np.random.rand(len(all_features))
	X_train_plus_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	
	# Take the matrix and sigmoid it with the weights - calculate error from
	# true (y_train) values.
	y_pred = log_model._sigmoid(np.dot(X_train_plus_bias, log_model.W))
	error = y_pred - y_train

	# Calculate manual and regular gradient.
	manual_gradient = np.dot(X_train_plus_bias.T, error) / len(y_train)
	gradient = log_model.calculate_gradient(y_train, X_train_plus_bias)

	# Confirm two matrices are equivalent...enough.
	assert np.allclose(gradient, manual_gradient)



def test_training():
	"""Confirm training improves the performance."""

	log_model.reset_model()
	random.seed(12)

	# Zero-shot predictions.
	dummy_y_pred = log_model.make_prediction(X_val)
	dummy_error = abs(dummy_y_pred - y_val)

	# Train the model.
	log_model.train_model(X_train, y_train, X_val, y_val)

	# Actual predictions.
	y_pred = log_model.make_prediction(X_val)
	error = abs(y_pred - y_val)

	# Most of the time the absolute error should be less often with the
	# trained model. Here it's better more than 90% of the time.
	assert sum(error < dummy_error) / len(error) > 0.9