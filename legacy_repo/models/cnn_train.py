import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

from keras.models import Sequential 
from keras.layers import Dense, Convolution1D, MaxPooling1D, Dropout, Flatten
import tensorflow as tf
from scaffold import Scaffold


with tf.device('/cpu:0'):
    def main():
        scaffold = Scaffold()
        args = scaffold.args

        cnn = Sequential() 

        (X_train,y_train,_,_) = scaffold.get_x_and_y()

        for i in range(0,args.number_of_layers):
            cnn.add(Convolution1D(args.number_of_neurons, 3,activation="relu",input_shape=(X_train.shape[1], 1)))
            cnn.add(MaxPooling1D(2))
            cnn.add(Dropout(0.1))
        cnn.add(Flatten())
        cnn.add(Dense(units=1,activation='sigmoid'))

        cnn.summary()

        cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy","f1_score","precision", "recall"])

        history = cnn.fit(X_train, y_train, epochs=args.number_of_epochs, batch_size=5000,validation_split=0.2)

        scaffold.save_model(cnn,"cnn")

    ###    test_results = ltsm.evaluate(X_test, y_test, verbose=1, return_dict=True)
    ###    print(test_results)

    if __name__ == '__main__':
        main()