#Creating a managed compute target with the SDK
#A managed compute target is one that is managed by Azure Machine Learning, such as an Azure Machine Learning compute cluster.

from azureml.core import Workspace
from azureml.core.compute import ComputeTarget,AmlCompute

ws=Workspace.from_config()

compute_name="aml-cluster"
compute_config=AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2',
                                                    min_node=0,max_nodes=4,
                                                    vm_priority='dedicated')

aml_cluster=ComputeTarget.creat(ws,compute_name,compute_config)
aml_cluster.wait_for_completion(show_output=True)



#Attaching an unmanaged compute target with the SDK
#An unmanaged compute target is one that is defined and managed outside of the Azure Machine Learning workspace;
# for example, an Azure virtual machine or an Azure Databricks cluster.

from azureml.core import Workspace
from azureml.core.compute import computeTarget,DatabricksCompute

ws=Workspace.from_config()

compute_name='db_cluster'


db_workspace_name='db_workspace'
db_resource_group='db_resource_group'
db_access_token='1234-23-ffg-ccf-...'
db_config=DatabricksCompute.attach_configuration(resource_group=db_resource_group,
                                                   workspace_name=db_workspace_name,
                                                   access_token=db_access_token)
databricks_compute=ComputeTarget.attach(ws,compute_name,db_config)
databricks_compute.wait_for_completion(True)

#Use compute targets
from azureml.core import Environment, ScriptRunConfig

compute_name = 'aml-cluster'

training_env = Environment.get(workspace=ws, name='training_environment')

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                environment=training_env,
                                compute_target=compute_name)

#Instead of specifying the name of the compute target, you can specify a ComputeTarget object, like this:



from azureml.core import Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget

compute_name = "aml-cluster"

training_cluster = ComputeTarget(workspace=ws, name=compute_name)

training_env = Environment.get(workspace=ws, name='training_environment')

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                environment=training_env,
                                compute_target=training_cluster)


#Checking for an existing compute target:
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

compute_name = "aml-cluster"

# Check if the compute target exists
try:
    aml_cluster = ComputeTarget(workspace=ws, name=compute_name)
    print('Found existing cluster.')
except ComputeTargetException:
    # If not, create it
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2',
                                                           max_nodes=4)
    aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)

aml_cluster.wait_for_completion(show_output=True)