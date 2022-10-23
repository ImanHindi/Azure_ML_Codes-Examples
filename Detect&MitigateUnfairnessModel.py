# Detect and Mitigate Unfairness in Models

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# load the diabetes dataset
print("Loading Data...")
data = pd.read_csv('/home/azureuser/cloudfiles/code/Users/2000117360/mslearn-dp100/data/diabetes.csv')

# Separate features and labels
features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
X, y = data[features].values, data['Diabetic'].values

# Get sensitive features
S = data[['Age']].astype(int)
# Change value to represent age groups
S['Age'] = np.where(S.Age > 50, 'Over 50', '50 or younger')

# Split data into training set and test set
X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, test_size=0.20, random_state=0, stratify=y)

# Train a classification model
print("Training model...")
diabetes_model = DecisionTreeClassifier().fit(X_train, y_train)

print("Model trained.")




from fairlearn.metrics import selection_rate, MetricFrame
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Get predictions for the witheld test data
y_hat = diabetes_model.predict(X_test)

# Get overall metrics
print("Overall Metrics:")
# Get selection rate from fairlearn
overall_selection_rate = selection_rate(y_test, y_hat) # Get selection rate from fairlearn
print("\tSelection Rate:", overall_selection_rate)
# Get standard metrics from scikit-learn
overall_accuracy = accuracy_score(y_test, y_hat)
print("\tAccuracy:", overall_accuracy)
overall_recall = recall_score(y_test, y_hat)
print("\tRecall:", overall_recall)
overall_precision = precision_score(y_test, y_hat)
print("\tPrecision:", overall_precision)

# Get metrics by sensitive group from fairlearn
print('\nMetrics by Group:')
metrics = {'selection_rate': selection_rate,
           'accuracy': accuracy_score,
           'recall': recall_score,
           'precision': precision_score}

group_metrics = MetricFrame(metrics=metrics,
                             y_true=y_test,
                             y_pred=y_hat,
                             sensitive_features=S_test['Age'])

print(group_metrics.by_group)



from raiwidgets import FairnessDashboard

# View this model in Fairlearn's fairness dashboard, and see the disparities which appear:
FairnessDashboard(sensitive_features=S_test,
                   y_true=y_test,
                   y_pred={"diabetes_model": diabetes_model.predict(X_test)})







# Separate features and labels
ageless = features.copy()
ageless.remove('Age')
X2, y2 = data[ageless].values, data['Diabetic'].values

# Split data into training set and test set
X_train2, X_test2, y_train2, y_test2, S_train2, S_test2 = train_test_split(X2, y2, S, test_size=0.20, random_state=0, stratify=y2)

# Train a classification model
print("Training model...")
ageless_model = DecisionTreeClassifier().fit(X_train2, y_train2)
print("Model trained.")

# View this model in Fairlearn's fairness dashboard, and see the disparities which appear:
FairnessDashboard(sensitive_features=S_test2,
                   y_true=y_test2,
                   y_pred={"ageless_diabetes_model": ageless_model.predict(X_test2)})




#Explore the model in the dashboard.
#
#When you review *recall*, note that the disparity has reduced, but the overall recall has also reduced because the model now significantly underpredicts positive cases for older patients. Even though **Age** was not a feature used in training, the model still exhibits some disparity in how well it predicts for older and younger patients.
#
#In this scenario, simply removing the **Age** feature slightly reduces the disparity in *recall*, but increases the disparity in *precision* and *accuracy*. This underlines one the key difficulties in applying fairness to machine learning models - you must be clear about what *fairness* means in a particular context, and optimize for that.
#
### Register the model and upload the dashboard data to your workspace
#
#You've trained the model and reviewed the dashboard locally in this notebook; but it might be useful to register the model in your Azure Machine Learning workspace and create an experiment to record the dashboard data so you can track and share your fairness analysis.
#
#Let's start by registering the original model (which included **Age** as a feature).
#
#**Note**: If you haven't already established an authenticated session with your Azure subscription, you'll be prompted to authenticate by clicking a link, entering an authentication code, and signing into Azure.



from azureml.core import Workspace, Experiment, Model
import joblib
import os

# Load the Azure ML workspace from the saved config file
ws = Workspace.from_config()
print('Ready to work with', ws.name)

# Save the trained model
model_file = 'diabetes_model.pkl'
joblib.dump(value=diabetes_model, filename=model_file)

# Register the model
print('Registering model...')
registered_model = Model.register(model_path=model_file,
                                  model_name='diabetes_classifier',
                                  workspace=ws)
model_id= registered_model.id


print('Model registered.', model_id)


from fairlearn.metrics._group_metric_set import _create_group_metric_set
from azureml.contrib.fairness import upload_dashboard_dictionary, download_dashboard_by_upload_id

#  Create a dictionary of model(s) you want to assess for fairness 
sf = { 'Age': S_test.Age}
ys_pred = { model_id:diabetes_model.predict(X_test) }
dash_dict = _create_group_metric_set(y_true=y_test,
                                    predictions=ys_pred,
                                    sensitive_features=sf,
                                    prediction_type='binary_classification')

exp = Experiment(ws, 'mslearn-diabetes-fairness')
print(exp)

run = exp.start_logging()

# Upload the dashboard to Azure Machine Learning
try:
    dashboard_title = "Fairness insights of Diabetes Classifier"
    upload_id = upload_dashboard_dictionary(run,
                                            dash_dict,
                                            dashboard_name=dashboard_title)
    print("\nUploaded to id: {0}\n".format(upload_id))

    # To test the dashboard, you can download it
    downloaded_dict = download_dashboard_by_upload_id(run, upload_id)
    print(downloaded_dict)
finally:
    run.complete()


from azureml.widgets import RunDetails

RunDetails(run).show()




#Mitigate unfairness in the model:
from fairlearn.reductions import GridSearch, EqualizedOdds
import joblib
import os

print('Finding mitigated models...')

# Train multiple models
sweep = GridSearch(DecisionTreeClassifier(),
                   constraints=EqualizedOdds(),
                   grid_size=20)

sweep.fit(X_train, y_train, sensitive_features=S_train.Age)
models = sweep.predictors_

# Save the models and get predictions from them (plus the original unmitigated one for comparison)
model_dir = 'mitigated_models'
os.makedirs(model_dir, exist_ok=True)
model_name = 'diabetes_unmitigated'
print(model_name)
joblib.dump(value=diabetes_model, filename=os.path.join(model_dir, '{0}.pkl'.format(model_name)))
predictions = {model_name: diabetes_model.predict(X_test)}
i = 0
for model in models:
    i += 1
    model_name = 'diabetes_mitigated_{0}'.format(i)
    print(model_name)
    joblib.dump(value=model, filename=os.path.join(model_dir, '{0}.pkl'.format(model_name)))
    predictions[model_name] = model.predict(X_test)


FairnessDashboard(sensitive_features=S_test,
                   y_true=y_test,
                   y_pred=predictions)


## Upload the mitigation dashboard metrics to Azure Machine Learning

As before, you might want to keep track of your mitigation experimentation. To do this, you can:

1. Register the models found by the GridSearch process.
2. Compute the performance and disparity metrics for the models.
3. Upload the metrics in an Azure Machine Learning experiment.

# Register the models
registered_model_predictions = dict()
for model_name, prediction_data in predictions.items():
    model_file = os.path.join(model_dir, model_name + ".pkl")
    registered_model = Model.register(model_path=model_file,
                                      model_name=model_name,
                                      workspace=ws)
    registered_model_predictions[registered_model.id] = prediction_data

#  Create a group metric set for binary classification based on the Age feature for all of the models
sf = { 'Age': S_test.Age}
dash_dict = _create_group_metric_set(y_true=y_test,
                                     predictions=registered_model_predictions,
                                     sensitive_features=sf,
                                     prediction_type='binary_classification')

exp = Experiment(ws, "mslearn-diabetes-fairness")
print(exp)

run = exp.start_logging()
RunDetails(run).show()

# Upload the dashboard to Azure Machine Learning
try:
    dashboard_title = "Fairness Comparison of Diabetes Models"
    upload_id = upload_dashboard_dictionary(run,
                                            dash_dict,
                                            dashboard_name=dashboard_title)
    print("\nUploaded to id: {0}\n".format(upload_id))
finally:
    run.complete()