# RSI-HFAS
《 RSI-HFAS：基于深度反馈机制的遥感图像连续比例超分辨率网络》

# Requirements

* Pytorch 1.1.0
* Python 3.6
* numpy
* skimage
* imageio
* cv2  


运行示例：

将model_400.pt放在./eperiment/metafpn/model/下

python main.py --model metafpn --ext sep  --save metafpn --n_GPUs 1 --batch_size 1 --test_only --data_test Set5 --pre_train  ./experiment/metafpn/model/model_400.pt  --save_results --scale 4.0

训练并测试：

python main.py --model metafpn --save metafpn --ext sep --lr_decay 90 --epochs 400 --n_GPUs 1 --batch_size 4
