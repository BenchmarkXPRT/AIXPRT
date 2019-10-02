from csv import DictReader
import os
import numpy as np
import mxnet as mx
import pickle 



CONTINUOUS_COLUMNS = ["I"+str(i) for i in range(1, 14)] # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C"+str(i) for i in range(1, 27)] # 1-26 inclusive
LABEL_COLUMN = ["clicked"]

TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
max_dict = {'I1': 1539, 'I2': 22066, 'I3': 65535, 'I4': 561, 'I5': 2655388, 'I6': 233523,
            'I7': 26297, 'I8': 5106, 'I9': 24376, 'I10': 9, 'I11': 181, 'I12': 1807, 'I13': 6879}
min_dict = {'I1': 0, 'I2': -3, 'I3': 0, 'I4': 0, 'I5': 0, 'I6': 0, 'I7': 0, 'I8': 0,
            'I9': 0, 'I10': 0, 'I11': 0, 'I12': 0, 'I13': 0}





def save_object(filename,obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def preprocess_uci_criteo(data_name):
    
    print("Preparing input dataset ... This could take some time.")
    hash_bucket_size = 1000
    #cont_defaults = [[0] for i in range(1, 14)]
    #cate_defaults = [[" "] for i in range(1, 27)]
    #label_defaults = [[0]]
    #column_headers = TRAIN_DATA_COLUMNS
    #record_defaults = label_defaults + cont_defaults + cate_defaults

    label_list = []
    csr_list = []
    dns_list = []

    #csr_ncols = len(CATEGORICAL_COLUMNS) * hash_bucket_size
    dns_ncols = len(CONTINUOUS_COLUMNS) + len(CATEGORICAL_COLUMNS)
    with open(data_name) as f:

        for row in DictReader(f, fieldnames=TRAIN_DATA_COLUMNS):
        
            label_list.append(row['clicked'])
            # Sparse base columns.
            for name in CATEGORICAL_COLUMNS:
                csr_list.append((hash(row[name]) % hash_bucket_size, 1.0))


            dns_row = [0] * dns_ncols
            dns_dim = 0
            # Embed wide columns into deep columns
            for col in CATEGORICAL_COLUMNS:
                dns_row[dns_dim] = hash(row[col].strip()) % hash_bucket_size
                dns_dim += 1
            # Continuous base columns.
            scale = 1 #align with Google WnD paper
            for col in CONTINUOUS_COLUMNS:
                #dns_row[dns_dim] = float(row[col].strip())
                orig_range = float(max_dict[col] - min_dict[col])
                dns_row[dns_dim] = (float(row[col].strip()) - min_dict[col]) * scale / orig_range
                dns_dim += 1
            # No transformations.

            dns_list.append(dns_row)
       
    data_list = [item[1] for item in csr_list]
    indices_list = [item[0] for item in csr_list]
    indptr_list = range(0, len(indices_list) + 1, len(CATEGORICAL_COLUMNS))
    csr = mx.nd.sparse.csr_matrix((data_list, indices_list, indptr_list),
                                  shape=(len(label_list), hash_bucket_size * len(CATEGORICAL_COLUMNS)))
    dns = np.array(dns_list)
    label = np.array(label_list)

    save_object('val_csr.pkl',csr)
    save_object('val_dns.pkl',dns)
    save_object('val_label.pkl',label)


if __name__ == "__main__":
    if not (os.path.exists(os.path.join(os.environ['APP_HOME'], "Modules", "Deep-Learning", "workloads","commonsources", "recommendation","val_csr.pkl"))):
        preprocess_uci_criteo('large_version/eval.csv')

