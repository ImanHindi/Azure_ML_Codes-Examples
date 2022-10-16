from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.runconfig import DockerConfiguration
from azureml.widgets import RunDetails

# Create a Python environment for the experiment (from a .yml file)
env = Environment.from_conda_specification("experiment_env", "/home/azureuser/cloudfiles/code/Users/2000117360/mslearn-dp100/environment.yml")

# Create a script config
script_config = ScriptRunConfig(source_directory=training_folder,
                                script='diabetes_training.py',
                                environment=env,
                                arguments = ['--reg_rate', 0.1],=
                                docker_runtime_config=DockerConfiguration(use_docker=True)) 

# submit the experiment run
experiment_name = 'mslearn-train-diabetes'
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)

# Show the running experiment run in the notebook widget
RunDetails(run).show()

# Block until the experiment run has completed
run.wait_for_completion()


# Get logged metrics and files
metrics = run.get_metrics()
for key in metrics.keys():
        print(key, metrics.get(key))
print('\n')
for file in run.get_file_names():
    print(file)




from azureml.core import Model

# Register the model
run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',
                   tags={'Training context':'Script'},
                   properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']})

# List registered models
for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')