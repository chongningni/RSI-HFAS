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

视觉结果：

![fig5_4](https://user-images.githubusercontent.com/58589797/236726207-66d12176-043f-4c18-bd93-0f8fa1abfbf3.png)


Citation:

@artical{ni2022hierarchical,

  author={Ni, Ning and Wu, Hanlin and Zhang, Libao},
  
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  
  title={Hierarchical Feature Aggregation and Self-Learning Network for Remote Sensing Image Continuous-Scale Super-Resolution}, 
  
  year={2022},
  
  volume={19},
  
  number={},
  
  pages={1-5},
  
  doi={10.1109/LGRS.2021.3122985}
  
 }
 
 持续更新中……………………………………………………………………………………
