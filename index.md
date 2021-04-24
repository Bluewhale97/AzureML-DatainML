## Introduction

Data is the most important matter in any machine learning workload, Azure provides datastores and datasets in an Azure ML workspace. In this article we will discuss about how to create and use datastores and datasets in Azure workspace.

In Azure Machine Learning, datastores are abstractions for cloud data sources. They encapsulate the information required to connect to data sources. You can access datastores directly in code by using the Azure Machine Learning SDK, and use it to upload or download data.

There are several types of datastores:

I think the classification of these types of datastores are depended on the file systems that we use, like SQL database, data lake, blob and file container as well as databricks file system.

![image](https://user-images.githubusercontent.com/71245576/115927676-cb4c9780-a452-11eb-8097-f097615fed5b.png)

One thing that should be noticed is that every workspace has two built-in datastores, an Azure Storage blob container, and an Azure Storage file container. We can add a third datastore to workspace as well. 

## 1. Using datastores

To add a datastore to workspace, there are two methods: use the GUI in Azure ML studio or use ML SDK. For example, the following code registers an Azure Storage blob container as a datastore named blob_data.

```python
from azureml.core import Workspace, Datastore

ws = Workspace.from_config()

# Register a new datastore
blob_ds = Datastore.register_azure_blob_container(workspace=ws, 
                                                  datastore_name='blob_data', 
                                                  container_name='data_container',
                                                  account_name='az_store_acct',
                                                  account_key='123456abcde789…')
```

You can view and manage datastores in Azure Machine Learning studio or use the SDK. For example, the following code lists the names of each datastore in the workspace.

```python
for ds_name in ws.datastores:
    print(ds_name)
```
It shows the names of my datastores:

![image](https://user-images.githubusercontent.com/71245576/115928314-e53aaa00-a453-11eb-9a7a-aa29ded68a7e.png)

You can get a reference to any datastore by using the Datastore.get() method as shown here:

```python
blob_store = Datastore.get(ws, datastore_name='blob_data')
```

The workspace always includes a default datastore (initially, this is the built-in workspaceblobstore datastore), which you can retrieve by using the get_default_datastore() method of a Workspace object, like this:

```python
default_store = ws.get_default_datastore()
```

It shows the infomation of the default datastore:

![image](https://user-images.githubusercontent.com/71245576/115928567-55493000-a454-11eb-90c9-d2dc957975fa.png)

There are several considreations that are deserved concerning about: 

![image](https://user-images.githubusercontent.com/71245576/115928695-8c1f4600-a454-11eb-8b05-009e651893c6.png)

When you want to change the default datastore, use the set_default_datastore() methods:

```python
ws.set_default_datastore('blob_data')
```

## 2. Using datasets

Datasets are versioned packaged data objects that can be easily consumed in experiments and pipelines. Datasets are the recommended way to work with data, and are the primary mechanism for advanced Azure Machine Learning capabilities like data labeling and data drift monitoring.

In Azure ML studio, there are types of dataset:

![image](https://user-images.githubusercontent.com/71245576/115928827-c688e300-a454-11eb-9b8e-b80b8c4d21e7.png)

You can use the visual interface in Azure Machine Learning studio or the Azure Machine Learning SDK to create datasets from individual files or multiple file paths. The paths can include wildcards (for example, /files/*.csv) making it possible to encapsulate data from a large number of files in a single dataset.

After you've created a dataset, you can register it in the workspace to make it available for use in experiments and data processing pipelines later.

Now let's look at how to create and register datasets using the SDK:

To create a tabular dataset using the SDK, use the from_delimited_files method of the Dataset.Tabular class, like this:

```python
from azureml.core import Dataset

blob_ds = ws.get_default_datastore()
csv_paths = [(blob_ds, 'data/files/current_data.csv'),
             (blob_ds, 'data/files/archive/*.csv')]
tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
tab_ds = tab_ds.register(workspace=ws, name='csv_table')
```

The dataset from two file paths within the default datastore, the current_data.csv file in the data/files folder and all .csv files in the data/files/archive/ folder.After creating the dataset, the code registers it in the workspace with the name csv_table.

You can retrieve a registered dataset by following techniques: the datasets dictionary attribute of a workspace object or the get_by_name or get_by_id method of the dataset class.

Like this:

```python
import azureml.core
from azureml.core import Workspace, Dataset

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Get a dataset from the workspace datasets collection
ds1 = ws.datasets['bike-rentals']

# Get a dataset by name from the datasets class
ds2 = Dataset.get_by_name(ws, 'bike-rentals')

ds1
ds2
```

![image](https://user-images.githubusercontent.com/71245576/115930488-8d05a700-a457-11eb-9914-ec8f894d8eac.png)

Datasets can be versioned, enabling you to track historical versions of datasets that were used in experiments, and reproduce those experiments with data in the same state.

You can create a new version of a dataset by registering it with the same name as a previously registered dataset and specifying the create_new_version property:

```python
img_paths = [(blob_ds, 'data/files/images/*.jpg'),
             (blob_ds, 'data/files/images/*.png')]
file_ds = Dataset.File.from_files(path=img_paths)
file_ds = file_ds.register(workspace=ws, name='img_files', create_new_version=True)
```

Retrieve a specific dataset version:

```python
img_ds = Dataset.get_by_name(workspace=ws, name='bike-rentals', version=1)
```
See the version info:

![image](https://user-images.githubusercontent.com/71245576/115932400-e15e5600-a45a-11eb-87aa-6805e2f39faa.png)

## 3. Working with tabular dataset

We can read data directly from a tabular dataset by converting it into a Pandas or Spark dataframe:

```python
df = tab_ds.to_pandas_dataframe()
# code to work with dataframe goes here, for example:
print(df.head())
```
When you need to access a dataset in an experiment script, you must pass the dataset to the script. There are two ways you can do this. The first is to use a script argument for a tabular dataset:

You can pass a tabular dataset as a script argument. When you take this approach, the argument received by the script is the unique ID for the dataset in your workspace. In the script, you can then get the workspace from the run context and use it to retrieve the dataset by it's ID.

The ScriptRunConfig is:

```python
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds', tab_ds],
                                environment=env)
```

The script script.py is:
```python
from azureml.core import Run, Dataset

parser.add_argument('--ds', type=str, dest='dataset_id')
args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace
dataset = Dataset.get_by_id(ws, id=args.dataset_id)
data = dataset.to_pandas_dataframe()
```

You also can use a named input for a tabular dataset. you can retrieve the dataset by name from the run context's input_datasets collection without needing to retrieve it from the workspace. Note that if you use this approach, you still need to include a script argument for the dataset, even though you don’t actually use it to retrieve the dataset.

The script:
```python
from azureml.core import Run

parser.add_argument('--ds', type=str, dest='ds_id')
args = parser.parse_args()

run = Run.get_context()
dataset = run.input_datasets['my_dataset']
data = dataset.to_pandas_dataframe()
```

## 4. Working with file datasets
When working with a file dataset, you can use the to_path() method to return a list of the file paths encapsulated by the dataset:
```path
for file_path in file_ds.to_path():
    print(file_path)
```
There are also two approaches to pass a file dataset to an experiment script, the first is to use a script argument for a file dataset, the second is to use a named input for a file dataset.

the ScriptRunConfig is:
```python
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds', file_ds.as_download()],
                                environment=env)
                                
```
See how to use a script argument for a file dataset:
```python
from azureml.core import Run
import glob

parser.add_argument('--ds', type=str, dest='ds_ref')
args = parser.parse_args()
run = Run.get_context()

imgs = glob.glob(ds_ref + "/*.jpg")
```
See how to use a named input for a file dataset:
```python
from azureml.core import Run
import glob

parser.add_argument('--ds', type=str, dest='ds_ref')
args = parser.parse_args()
run = Run.get_context()

dataset = run.input_datasets['my_ds']
imgs= glob.glob(dataset + "/*.jpg")
```


Reference:

Build AI solutions with Azure Machine Learning, retrieved from https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/
