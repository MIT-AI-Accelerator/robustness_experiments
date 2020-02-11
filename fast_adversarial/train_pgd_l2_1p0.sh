PYTHONPATH=$HOME/python/robustness python -m robustness.main --dataset cifar \
--data data/cifar10 --arch resnet50 \
--adv-train 1 --constraint 2 \
--attack-steps 20 --eps 1.0 --attack-lr 0.125 \
--out-dir logs --exp-name pgd_1p0
