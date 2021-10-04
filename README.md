### Introduction  
**Yolort implemented using ppl.nn [https://github.com/zhiqwang/yolov5-rt-stack](https://github.com/zhiqwang/yolov5-rt-stack)**

### Usage
- **Build ppl.nn**
```
https://github.com/openppl-public/ppl.nn.git
cd ppl.nn
./build.sh -DHPCC_USE_X86_64=ON -DPPLNN_ENABLE_PYTHON_API=ON

cp -r ./pplnn-build/install/* ./third_party
```
- **Build CODE**
```
mkdir build
cd build
cmake ..
make -j 4
```

- **Test**
```
./test_yolov5 ./assert/yolov5_sim.onnx ./assert/bus.jpg
```
