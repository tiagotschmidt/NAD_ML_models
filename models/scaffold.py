import argparse
from os import path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import model_from_json # saving and loading trained model


class Scaffold():
    args = {}
    MAX_FEATURES = 93
    def __init__(self):
        parser = argparse.ArgumentParser(description="MLP NAD")
        parser.add_argument('--number-of-layers', type=int, default=1, metavar='N', help="Number of layers used, excluding final layer (default=1)")
        parser.add_argument('--number-of-neurons', type=int, default=50, metavar='N', help="Number of neurons per layer used (default=50)")
        parser.add_argument('--number-of-epochs', type=int, default=100, metavar='N', help="Number of epochs (default=100)")
        parser.add_argument('--number-of-features-removed', type=int, default=0, metavar='N', help="Number of features removed from model (default=0)")
        self.args = parser.parse_args()
        print(self.args)

    def get_x_and_y(self):
        preprocessed_data = pd.read_csv('dataset/preprocessed_binary_dataset.csv')

        features = preprocessed_data.iloc[:,0:self.MAX_FEATURES-self.args.number_of_features_removed].values 
        target = preprocessed_data[['intrusion']].values 

        X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.25, random_state=42)
        X_train=np.asarray(X_train).astype(np.float32)
        X_test=np.asarray(X_test).astype(np.float32)
        return (X_train,y_train,X_test,y_test)

    def get_full_dataset(self):
        preprocessed_data = pd.read_csv('dataset/preprocessed_binary_dataset.csv')
        features = preprocessed_data.iloc[:,0:self.MAX_FEATURES-self.args.number_of_features_removed].values 
        target = preprocessed_data[['intrusion']].values 

        X_train=np.asarray(features).astype(np.float32)
        X_test=np.asarray(target).astype(np.float32)
        return (X_train,X_test)

    def save_model(self, model):
        filepath = f"./models/json_models/mlp_{self.args.number_of_layers}_{self.args.number_of_neurons}_{self.args.number_of_epochs}_{self.args.number_of_features_removed}.json"
        weightspath = f"./models/models_weights/mlp_{self.args.number_of_layers}_{self.args.number_of_neurons}_{self.args.number_of_epochs}_{self.args.number_of_features_removed}.weights.h5"
        if (not path.isfile(filepath)):
            mlp_json = model.to_json()
            with open(filepath, "w") as json_file:
                json_file.write(mlp_json)

            model.save_weights(weightspath)

    def load_model(self,model_name:str):
        filepath = f"./models/json_models/{model_name}_{self.args.number_of_layers}_{self.args.number_of_neurons}_{self.args.number_of_epochs}_{self.args.number_of_features_removed}.json"
        weightspath = f"./models/models_weights/{model_name}_{self.args.number_of_layers}_{self.args.number_of_neurons}_{self.args.number_of_epochs}_{self.args.number_of_features_removed}.weights.h5"
        json_file = open(filepath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(weightspath)
        return model
