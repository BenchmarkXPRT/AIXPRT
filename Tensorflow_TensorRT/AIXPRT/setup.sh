git clone https://github.com/tensorflow/models.git
cd models
export PYTHONPATH="$PYTHONPATH:$PWD"
cd research
export PYTHONPATH="$PYTHONPATH:$PWD"
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.
cd slim
python setup.py install
pip install requests pillow
