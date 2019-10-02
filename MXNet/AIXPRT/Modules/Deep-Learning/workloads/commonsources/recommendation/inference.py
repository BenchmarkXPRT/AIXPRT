"""inference script to support accuracy and performance benchmark"""
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import logging
import ctypes
import time
import os
import pickle
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.base import check_call, _LIB



def load_model(_symbol_file, _param_file):
    """load existing symbol model"""
    cur_path = os.path.dirname(os.path.realpath(__file__))
    symbol_file_path = os.path.join(cur_path, _symbol_file)

    symbol = mx.sym.load(symbol_file_path)

    param_file_path = os.path.join(cur_path, _param_file)

    save_dict = nd.load(param_file_path)
    _arg_params = {}
    _aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            _arg_params[name] = v
        if tp == 'aux':
            _aux_params[name] = v
    return symbol, _arg_params, _aux_params

def advance_data_iter(data_iter, n):
    """use to warm up data for performance benchmark"""
    assert n >= 0
    if n == 0:
        return data_iter
    has_next_batch = True
    while has_next_batch:
        try:
            data_iter.next()
            n -= 1
            if n == 0:
                return data_iter
        except StopIteration:
            has_next_batch = False

CRITEO = {
    'train': 'train.csv',
    'test': 'eval.csv',
    'num_linear_features': 26000,
    'num_embed_features': 26,
    'num_cont_features': 13,
    'embed_input_dims': 1000,
    'hidden_units': [32, 1024, 512, 256],
}
def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a model on a dataset')
    parser.add_argument('--aarch', type=str, default='')
    parser.add_argument('--symbol-file', type=str, default='checkpoint-symbol.json', help='symbol file path')
    parser.add_argument('--param-file', type=str, default='checkpoint-0000.params', help='param file path')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--csv-file-path', type=str, default='')
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--shuffle-dataset', action='store_true', default=True,
                        help='shuffle the calibration dataset')
    parser.add_argument('--num-omp-threads', type=int, default=28)
    parser.add_argument('--num-batches', type=int, default=8000000)
    parser.add_argument('--iterations', type=int)
    args = parser.parse_args()

    if args.aarch == 'CPU':
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(0)




    symbol_file = args.symbol_file
    param_file = args.param_file

    # timing_csv_file = open(args.csv_file_path,"ab")
    batch_size = args.batch_size
    label_name = args.label_name

    val_csr = load_object('val_csr.pkl')
    val_dns = load_object('val_dns.pkl')
    val_label = load_object('val_label.pkl')


    # creating data iterator
    data = mx.io.NDArrayIter({'csr_data': val_csr, 'dns_data': val_dns},
                             {'softmax_label': val_label}, batch_size,
                             shuffle=False, last_batch_handle='discard')

    # loading model
    sym, arg_params, aux_params = load_model(symbol_file, param_file)


    # make sure that fp32 inference works on the same images as calibrated quantized model


    acc_m = mx.metric.create('acc')
    mod = mx.mod.Module(symbol=sym, context=ctx, data_names=['csr_data', 'dns_data'], label_names=[label_name, ])
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)

    check_call(_LIB.MXSetNumOMPThreads(ctypes.c_int(args.num_omp_threads)))
    batch_data = []
    nbatch = 0
    for batch in data:
        if nbatch < args.num_batches:
            batch_data.append(batch)
            nbatch += 1
        else:
            break
    #for data warmup
    wi = 50
    i = 0
    for batch in batch_data:
        if i < wi:
            mod.forward(batch, is_train=False)
            i += 1
        else:
            break
    data.hard_reset()
    mx.nd.waitall()
    #real run

    nbatch = 0
    timing_csv_buffer_data = [] 
    for batch in batch_data:
        nbatch += 1
        start = time.time()
        mod.forward(batch, is_train=False)
        mx.nd.waitall()
        end = time.time()
        timing_csv_buffer_data.append(str(start) + "," + str(end)) 



        if(nbatch>=args.iterations):
            break



    with open(args.csv_file_path,"w") as txt_file:
        for line in timing_csv_buffer_data:
            txt_file.write(line + "\n") 

