# Predicting Task-relevant Objects in Egocentric Videos

This is the implementation of Predicting Task-relevant Objects in Egocentric Videos. 
Please visit the <a href="https://luohwu.github.io/">project page</a> for details.

## Environment
The models were implemented with Python 3.8.10 and Pytorch 1.10.0 on Ubuntu 20.04. For installing
other packages, please run `pip install -r requirements.txt`.



## Dataset
We use the Short-term Human-object Interaction Anticipation benchmark of
<a href="https://ego4d-data.org/#download">EGO4D</a> dataset. Please follow
this <a href="https://ego4d-data.org/docs/start-here/"> guideline </a> 
for accessing it. In our implementation, all frames are resized to 456x256.

## EgoMotionNet 
EgoMotionNet serves as the backbone to extract temporal context in 
IntentNet. 
For easier usage, we train and evaluate it in another python 
<a href="https://github.com/luohwu/EgoMotionNet">project</a>.
## Train
``python train_ambiguity_clip_word2vec.py --data_path **/**/dataset/EGO4D/ --exp_path ***/***/experiments
--exp_name clip --lr 3e-3 -bs 32 epoch 1000``
<br>
explanations:
<ul>
<li>data_path: path to folder of dataset</li>
<li>exp_path: path to folder of experiments</li>
<li>exp_name: specific name for the current experiment</li>
<li>lr: learning rate</li>
<li>bs: batch size</li>
<li>epoch: number of training epochs</li>
</ul>


## Evaluation
``python eval_ambiguity_clip.py --data_path **/**/dataset/EGO4D/ --exp_path ***/***/experiments
--exp_name clip --model_path **/**/model_epoch_xx.pth``
<br>
explanations:
<ul>
<li>data_path: path to folder of dataset</li>
<li>exp_path: path to folder of experiments</li>
<li>exp_name: specific name for the current experiment</li>
<li>model_path: path to pre-trained model</li>
</ul>

Pretrained models are available in `./pre-trained_models/`
