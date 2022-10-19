from azureml.core import Workspace, Datastore

ws = Workspace.from_config()

# Register a new datastore
blob_ds = Datastore.register_azure_blob_container(workspace=ws, 
                                                  datastore_name='blob_data', 
                                                  container_name='data_container',
                                                  account_name='az_store_acct',
                                                  account_key='123456abcde789…')



#the following code lists the names of each datastore in the workspace.
for ds_name in ws.datastores:
    print(ds_name)

#get a reference to any datastore by using the Datastore.get() method

blob_store = Datastore.get(ws, datastore_name='blob_data')


# a default datastore (initially, this is the built-in workspaceblobstore datastore),
#  which you can retrieve by using the get_default_datastore()
#  method of a Workspace object, like this:
default_store = ws.get_default_datastore()

#To change the default datastore, use the set_default_datastore() method:

ws.set_default_datastore('blob_data')


#Creating and registering tabular datasets 
from azure.core import Dataset
blob_ds = ws.get_default_datastore()
csv_paths=[(blob_ds,'data/files/current_data.csv'),
            (blob_ds,'data/files/archive/*.csv')]
tab_ds=Dataset.Tabular.from_delimited_files(path=csv_paths)
tab_ds=tab_ds.register(Workspace=ws,name='csv_table')

#Use datasets
#Work with tabular datasets
df = tab_ds.to_pandas_dataframe()
# code to work with dataframe goes here, for example:
print(df.head())






#Creating and registering file datasets
from azureml.core import Dataset

blob_ds = ws.get_default_datastore()
file_ds = Dataset.File.from_files(path=(blob_ds, 'data/files/images/*.jpg'))
file_ds = file_ds.register(workspace=ws, name='img_files')



#Retrieving a registered dataset
#1.The datasets dictionary attribute of a Workspace object.
#2.The get_by_name or get_by_id method of the Dataset class.

import azureml.core
from azureml.core import Workspace, Dataset

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Get a dataset from the workspace datasets collection
ds1 = ws.datasets['csv_table']

# Get a dataset by name from the datasets class
ds2 = Dataset.get_by_name(ws, 'img_files')



#Dataset versioning
img_paths = [(blob_ds, 'data/files/images/*.jpg'),
             (blob_ds, 'data/files/images/*.png')]
file_ds = Dataset.File.from_files(path=img_paths)
file_ds = file_ds.register(workspace=ws, name='img_files', create_new_version=True)


#Retrieving a specific dataset version
img_ds = Dataset.get_by_name(workspace=ws, name='img_files', version=2)



#Pass a tabular dataset to an experiment script:
#1.Use a script argument for a tabular dataset(retrieve the dataset by it's ID):

#ScriptRunConfig:
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds', tab_ds],
                                environment=env)

#Script:
from azureml.core import Run, Dataset

parser.add_argument('--ds', type=str, dest='dataset_id')
args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace
dataset = Dataset.get_by_id(ws, id=args.dataset_id)
data = dataset.to_pandas_dataframe()


#2.Use a named input for a tabular dataset:
#ScriptRunConfig:

env=Environment('my_env')
packages=CondaDependencies.create(conda_packages=['pip'],pip_packages=['azureml-defaults',
                                                                        'azureml-dataprep[pandas]'])

env.python.conda_dependencies=packages

script_config=ScripRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds',tab_ds.as_named_input("my_dataset"),
                                environment=env])


#script:

from azureml.core import Run
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--ds', type=str, dest='ds_id')
args = parser.parse_args()

run=Run.getcontext()
dataset=run.input_datasets['my_dataset']
data=dtaset.to_pandas_dataframe()



#Work with file datasets

for file_path in file_ds.to_path():
    print(file_path)

#Pass a file dataset to an experiment script:
#1.Use a script argument for a file dataset
script_config=ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments['--ds',file_ds.as_download()],
                                environment=env)

#script:
from azureml.core import Run
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--ds',type=str,dest='ds_ref')
args=parser.parse_args()
reun=Run.get_context()
imgs=glob.glob(args.ds_ref+"/*.jpg")

#2.Use a named input for a file dataset:


#ScriptRunConfig:
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds', file_ds.as_named_input('my_ds').as_download()],
                                environment=env)


#script:
from azureml.core import Run
import glob

parser.add_argument('--ds', type=str, dest='ds_ref')
args = parser.parse_args()
run = Run.get_context()

dataset = run.input_datasets['my_ds']
imgs= glob.glob(dataset + "/*.jpg")
