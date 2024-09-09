import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

from keras.models import Sequential 
from keras.layers import Dense,Flatten,LSTM 
from scaffold import Scaffold


def main():
    scaffold = Scaffold()
    args = scaffold.args

    ltsm = Sequential() 
    
    (X_train,y_train,X_test,y_test) = scaffold.get_x_and_y()

    for i in range(0,args.number_of_layers):
        ltsm.add(LSTM(units=args.number_of_neurons, return_sequences=True,input_shape=(X_train.shape[1],1)))
    ltsm.add(Flatten())
    ltsm.add(Dense(units=1,activation='sigmoid'))

    ltsm.summary()

    ltsm.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy","f1_score","precision", "recall"])

    history = ltsm.fit(X_train, y_train, epochs=args.number_of_epochs, batch_size=5000,validation_split=0.2)

    scaffold.save_model(ltsm,"lstm")

###    test_results = ltsm.evaluate(X_test, y_test, verbose=1, return_dict=True)
###    print(test_results)

if __name__ == '__main__':
    main()