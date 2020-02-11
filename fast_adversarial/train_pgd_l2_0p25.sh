PYTHONPATH=$HOME/python/robustness python -m robustness.main --dataset cifar \
--data data/cifar10 --arch resnet50 \
--adv-train 1 --constraint 2 \
--attack-steps 20 --eps 0.25 --attack-lr 0.03125 \
--out-dir logs --exp-name pgd_0p25
