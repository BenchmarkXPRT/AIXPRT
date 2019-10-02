#!/bin/sh


cd $1
if [ -d $1/large_version ]
then
    echo "Data Exist. Executing workload"
else
    echo "Downloading Data.."
    wget -P large_version https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv
fi

python3 pkGen.py

mv *.pkl  ../../commonsources/recommendation/




