import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

from scaffold import Scaffold

def main():
    scaffold = Scaffold()
    args = scaffold.args

    lstm = scaffold.load_model("lstm")
    (X_train,y_train) = scaffold.get_full_dataset()
    lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy","f1_score","precision", "recall"])

    for i in range (0,100):
        test_results = lstm.evaluate(X_train, y_train, verbose=1, return_dict=True)
        print(test_results)

if __name__ == '__main__':
    main()