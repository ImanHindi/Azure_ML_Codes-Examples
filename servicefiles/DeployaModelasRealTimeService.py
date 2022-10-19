#Deploy a model as a real-time service
from azureml.core import Model

classification_model = Model.register(workspace=ws,
                       model_name='classification_model',
                       model_path='model.pkl', # local path
                       description='A classification model')

#Alternatively, if you have a reference to the Run used to train the model,
#  you can use its register_model method as shown here:

run.register_model( model_name='classification_model',
                    model_path='outputs/model.pkl', # run outputs path
                    description='A classification model')




#Create an environment

from azureml.core import Environment

service_env = Environment(name='service-env')
python_packages = ['scikit-learn', 'numpy'] # whatever packages your entry script uses
for package in python_packages:
    service_env.python.conda_dependencies.add_pip_package(package)


#Combine the script and environment in an InferenceConfig
from azureml.core.model import InferenceConfig

classifier_inference_config = InferenceConfig(source_directory = 'service_files',
                                              entry_script="score.py",
                                              environment=service_env)
#create AKS compute target configurations:
from azureml.core.compute import ComputeTarget, AksCompute

cluster_name = 'aks-cluster'
compute_config = AksCompute.provisioning_configuration(location='eastus')
production_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
production_cluster.wait_for_completion(show_output=True)

#3. Define a deployment configuration
# configure the compute to which the service will be deployed. If you are deploying to an AKS cluster
from azureml.core.compute import ComputeTarget, AksCompute

cluster_name = 'aks-cluster'
compute_config = AksCompute.provisioning_configuration(location='eastus')
production_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
production_cluster.wait_for_completion(show_output=True)


#define the deployment configuration, which sets the target-specific compute specification for the containerized deployment:

from azureml.core.webservice import AksWebservice

classifier_deploy_config = AksWebservice.deploy_configuration(cpu_cores = 1,
                                                              memory_gb = 1)

#The code to configure an ACI deployment is similar,
#  except that you do not need to explicitly create an ACI compute target, 
# and you must use the deploy_configuration class from the azureml.core.webservice.AciWebservice namespace.
#  Similarly, you can use the azureml.core.webservice.LocalWebservice
#  namespace to configure a local Docker-based service.


#4. Deploy the model
from azureml.core.model import Model

model = ws.models['classification_model']
service = Model.deploy(workspace=ws,
                       name = 'classifier-service',
                       models = [model],
                       inference_config = classifier_inference_config,
                       deployment_config = classifier_deploy_config,
                       deployment_target = production_cluster)
service.wait_for_deployment(show_output = True)

#For ACI or local services, you can omit the deployment_target parameter (or set it to None).


#Consume a real-time inferencing service

import json

# An array of new data cases
x_new = [[0.1,2.3,4.1,2.0],
         [0.2,1.8,3.9,2.1]]

# Convert the array to a serializable list in a JSON document
json_data = json.dumps({"data": x_new})

# Call the web service, passing the input data
response = service.run(input_data = json_data)

# Get the predictions
predictions = json.loads(response)

# Print the predicted class for each case.
for i in range(len(x_new)):
    print (x_new[i], predictions[i])


#Troubleshoot service deployment
from azureml.core.webservice import AksWebservice

# Get the deployed service
service = AksWebservice(name='classifier-service', workspace=ws)

# Check its state
#To view the state of a service, you must use the compute-specific service type
#(for example AksWebservice) and not a generic WebService object.
print(service.state)


#Review service logs
print(service.get_logs())



#Deploy to a local container
from azureml.core.webservice import LocalWebservice

deployment_config = LocalWebservice.deploy_configuration(port=8890)
service = Model.deploy(ws, 'test-svc', [model], inference_config, deployment_config)

print(service.run(input_data = json_data))

#You can then troubleshoot runtime issues by making changes to the scoring file that is referenced
#  in the inference configuration, and reloading the service without redeploying it 
# (something you can only do with a local service):
service.reload()
print(service.run(input_data = json_data))