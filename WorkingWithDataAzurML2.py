import azureml.core
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))



# Get the default datastore
default_ds = ws.get_default_datastore()

# Enumerate all datastores, indicating which is the default
for ds_name in ws.datastores:
    print(ds_name, "- Default =", ds_name == default_ds.name)


from azureml.core import Dataset
from azureml.data.datapath import DataPath

Dataset.File.upload_directory(src_dir='/home/azureuser/cloudfiles/code/Users/2000117360/mslearn-dp100/data',
                              target=DataPath(default_ds, '/home/azureuser/cloudfiles/code/diabetes-data/')
                              )


from azureml.core import Dataset

# Get the default datastore
default_ds = ws.get_default_datastore()

#Create a tabular dataset from the path on the datastore (this may take a short while)
tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, '/home/azureuser/cloudfiles/code/diabetes-data/*.csv'))

# Display the first 20 rows as a Pandas dataframe
tab_data_set.take(20).to_pandas_dataframe()




#Create a file dataset from the path on the datastore (this may take a short while)
file_data_set = Dataset.File.from_files(path=(default_ds, '/home/azureuser/cloudfiles/code/diabetes-data/*.csv'))

# Get the files in the dataset
for file_path in file_data_set.to_path():
    print(file_path)




# Register the tabular dataset
try:
    tab_data_set = tab_data_set.register(workspace=ws, 
                                        name='diabetes dataset',
                                        description='diabetes data',
                                        tags = {'format':'CSV'},
                                        create_new_version=True)
except Exception as ex:
    print(ex)

# Register the file dataset
try:
    file_data_set = file_data_set.register(workspace=ws,
                                            name='diabetes file dataset',
                                            description='diabetes files',
                                            tags = {'format':'CSV'},
                                            create_new_version=True)
except Exception as ex:
    print(ex)

print('Datasets registered')



print("Datasets:")
for dataset_name in list(ws.datasets.keys()):
    dataset = Dataset.get_by_name(ws, dataset_name)
    print("\t", dataset.name, 'version', dataset.version)