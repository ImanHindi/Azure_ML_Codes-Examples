#Running automated machine learning experiments
#Configure an automated machine learning experiment
from azureml.train.automl import AutoMLConfig

automl_run_config = RunConfiguration(framework='python')
automl_config = AutoMLConfig(name='Automated ML Experiment',
                             task='classification',
                             primary_metric = 'AUC_weighted',
                             compute_target=aml_compute,
                             training_data = train_dataset,
                             validation_data = test_dataset,
                             label_column_name='Label',
                             featurization='auto',
                             iterations=12,
                             max_concurrent_iterations=4)

from azureml.train.automl.utilities import get_primary_metrics


#o retrieve the list of metrics available for a particular task type,
#  you can use the get_primary_metrics function as shown here:
get_primary_metrics('classification')


#Submit an automated machine learning experiment
from azureml.core.experiment import Experiment

automl_experiment = Experiment(ws, 'automl_experiment')
automl_run = automl_experiment.submit(automl_config)


#You can monitor automated machine learning experiment runs in Azure Machine Learning studio,
#  or in the Jupyter Notebooks RunDetails widget
#Retrieve the best run and its model
best_run, fitted_model = automl_run.get_output()
best_run_metrics = best_run.get_metrics()
for metric_name in best_run_metrics:
    metric = best_run_metrics[metric_name]
    print(metric_name, metric)

#Explore preprocessing steps
for step_ in fitted_model.named_steps:
    print(step_)