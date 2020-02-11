from FastAdversarial import main
from cox.utils import Parameters

from robustness.main import setup_args, setup_store_with_metadata

EPS = 0.25
eps_str = f'{EPS}'.replace('.', 'p')
exp_name = f'fast_adv_l2_{eps_str}'

train_kwargs = {
    'out_dir': "logs",
    'exp_name': exp_name,
    'dataset': 'cifar',
    'data': 'data/cifar10',
    'arch': 'resnet50',
    'adv_train': 1,
    'constraint': '2',
    'eps': EPS,
    'attack_steps': 20,
    'attack_lr': 2.5 * EPS / 20,
    'fast_attack_lr': 1.5 * EPS
}
train_args = Parameters(train_kwargs)

# Fill whatever parameters are missing from the defaults
train_args = setup_args(train_args)
print(train_args)
store = setup_store_with_metadata(train_args)

final_model = main(train_args, store=store)

