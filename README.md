### PAPER

This is a pytorch implementation of [Feedback-Network](http://feedbacknet.stanford.edu/) (CVPR 2017, Zamir et al.)

### Requirements

- Pytorch = 0.3.1
- python = 2.7
- numpy >= 1.14.2

### Train

- Data：
  - Download [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset (binary file), and put it under folder `./data/`

- Run：
  - Run `classifier_train.py`

  ```python
  python FeedbackNet_train.py
  ```

  - The trained model will saved in folder `./models/` every 10 epochs。
  - Attention：
    - You can adjust the parameter `batch_size` to fit your GPU memory。


### Evaluation

- Data：
  - Download [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset (binary file), and put it under folder `./data/`


- Evaluate on CIFAR100 dataset：

  - Put the checkpoint files under folder `./models/`, and specify the checkpoints file path by setting parameter `params.ckpt` in file `classifier_train.py`. You can download my baseline model [here](https://cloud.tsinghua.edu.cn/f/8b1affe99ba5494ba636/)
  - Run `classifier_train.py`​

  ```python
  python FeedbackNet_test.py
  ```

### Result

- Val Accuracy

| Physical / Virtual Depth | PAPER(tp1) |  PAPER(tp5)  | ME(tp1) |  ME(tp5)  |
| :------:   | :----:   | :----: | :----: | :----: |
| 12 / 48 (stack-3;iteration=4) | 71.12%|91.51%|70.92%|92.02%|

### To Do

- [ ] Skip connections
- [ ] Multi-GPU
- [ ] Other virtual depths

### Reference

- [@amir32002 feedback-networks](https://github.com/amir32002/feedback-networks) : The Torch7 repo of Feedback Networks, Zamir et al.
- Thanks [@maxspero](https://github.com/maxspero/feedback-networks-pytorch) for the implemention of `FeedbackNet`. In this repository, I change a few codes in `./network/feedbacknet.py` and `./network/convlstmstack.py` to fit the model definition in the [original paper](http://feedbacknet.stanford.edu/)
- [@bzcheeseman_pytorch-feedbacknet](https://github.com/bzcheeseman/pytorch-feedbacknet) 