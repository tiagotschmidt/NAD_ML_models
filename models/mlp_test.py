import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

from scaffold import Scaffold

def main():
    scaffold = Scaffold()
    args = scaffold.args

    mlp = scaffold.load_model("mlp")
    (X_train,y_train) = scaffold.get_full_dataset()
    mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    for i in range (0,100):
        test_results = mlp.evaluate(X_train, y_train, verbose=1, return_dict=True)
        print(test_results)

if __name__ == '__main__':
    main()