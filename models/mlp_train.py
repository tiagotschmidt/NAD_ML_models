import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

from keras.models import Sequential 
from keras.layers import Dense 
from scaffold import Scaffold


def main():
    scaffold = Scaffold()
    args = scaffold.args

    mlp = Sequential() # creating model
    
    (X_train,y_train,X_test,y_test) = scaffold.get_x_and_y()

    for i in range(0,args.number_of_layers):
        # adding input layer and first layer with 50 neurons
        mlp.add(Dense(units=args.number_of_neurons, input_dim=X_train.shape[1], activation='relu'))
    # output layer with sigmoid activation
    mlp.add(Dense(units=1,activation='sigmoid'))

    mlp.summary()

    mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy","f1_score","precision", "recall"])

    history = mlp.fit(X_train, y_train, epochs=args.number_of_epochs, batch_size=5000,validation_split=0.2)

    scaffold.save_model(mlp,"mlp")

###    test_results = mlp.evaluate(X_test, y_test, verbose=1, return_dict=True)
###    print(test_results)

if __name__ == '__main__':
    main()