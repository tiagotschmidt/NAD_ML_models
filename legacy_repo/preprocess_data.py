import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


def map_attack_labels_into_categories(df):
    df.label.replace(
        [
            "apache2",
            "back",
            "land",
            "neptune",
            "mailbomb",
            "pod",
            "processtable",
            "smurf",
            "teardrop",
            "udpstorm",
            "worm",
        ],
        "Dos",
        inplace=True,
    )
    df.label.replace(
        [
            "ftp_write",
            "guess_passwd",
            "httptunnel",
            "imap",
            "multihop",
            "named",
            "phf",
            "sendmail",
            "snmpgetattack",
            "snmpguess",
            "spy",
            "warezclient",
            "warezmaster",
            "xlock",
            "xsnoop",
        ],
        "R2L",
        inplace=True,
    )
    df.label.replace(
        ["ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"],
        "Probe",
        inplace=True,
    )
    df.label.replace(
        [
            "buffer_overflow",
            "loadmodule",
            "perl",
            "ps",
            "rootkit",
            "sqlattack",
            "xterm",
        ],
        "U2R",
        inplace=True,
    )


def normalize_numeric_column(df, col):
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = standard_scaler.fit_transform(arr.reshape(len(arr), 1))
    return df


col_names = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty_level",
]


### Read the original KDD NSL dataset file.
original_data = pd.read_csv("dataset/KDDTrain+.txt", header=None, names=col_names)

### Drop useless feature
original_data.drop(["difficulty_level"], axis=1, inplace=True)

map_attack_labels_into_categories(original_data)

### Select and normalize numeric columns
numeric_columns = original_data.select_dtypes(include="number").columns
standard_scaler = StandardScaler()
original_data = normalize_numeric_column(original_data.copy(), numeric_columns)

### OneHotEncode categorical columns
categorical_columns = ["protocol_type", "service", "flag"]
categorical_data = original_data[categorical_columns]
categorical_data = pd.get_dummies(categorical_data, columns=categorical_columns)

### Map original data label's into normal or abnormal
normal_abnormal_column = pd.DataFrame(
    original_data.label.map(lambda x: "normal" if x == "normal" else "abnormal")
)
normal_abnormal_data = original_data.copy()
normal_abnormal_data["label"] = normal_abnormal_column

### OneHotEncode Normal/Abnormal label
le1 = preprocessing.LabelEncoder()
one_hot_encoded_normal_abnormal_column = normal_abnormal_column.apply(le1.fit_transform)
normal_abnormal_data["intrusion"] = one_hot_encoded_normal_abnormal_column

### Select the numerical data from original dataset
numerical_data = normal_abnormal_data[numeric_columns]
### Add the instrusion (target) column
numerical_data["intrusion"] = normal_abnormal_data["intrusion"]
### Filter more interesting features from the numerical feature pool.
numerical_data = normal_abnormal_data[
    [
        "count",
        "srv_serror_rate",
        "serror_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "logged_in",
        "dst_host_same_srv_rate",
        "dst_host_srv_count",
        "same_srv_rate",
    ]
]
### Join the categorical (one hot encoded)  data.
numerical_data = numerical_data.join(categorical_data)
### Join selected numerical, encoded categorical and the target label and rewrites the final data used.
preprocessed_data = numerical_data.join(normal_abnormal_data["intrusion"])
preprocessed_data.to_csv("./dataset/preprocessed_binary_dataset.csv")
