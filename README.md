# Irony Detection
Insert project description...

# File Structure
There are two main files, train.py and validate.py. Each requires a configuration file to be passed as the argument.

Train.py will first read the configuration file (use config_fer.json as an example). Then, it will load the dataset loader (which loads the data) from the 'datasets' folder with the name *config_dataset_name*_dataset.py. Once the dataset loader is loaded, train.py will load the data using this loader. Then, the model will be loaded from the 'models' folder with name *config_model_name*_model.py. Then, training will begin!

These dataset loader and model folders are what will be created.
