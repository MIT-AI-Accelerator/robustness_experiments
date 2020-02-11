import os

from FastAdversarial import main
from cox.utils import Parameters
from robustness.main import setup_args, setup_store_with_metadata

EPS = 0.25
eps_str = f'{EPS}'.replace('.', 'p')
exp_name = f'fast_adv_l2_{eps_str}'

for e in [0, 0.25, 0.5, 1.0, 2.0]:
    e_str = f'{e}'.replace('.', 'p')
    eval_exp_name = f'fast_adv_l2_{eps_str}_{e_str}'

    eval_kwargs = {
        'eval_only': 1,
        'resume' : os.path.join('logs', exp_name, 'checkpoint.pt.best'),
        'out_dir': "logs_eval",
        'exp_name': eval_exp_name,
        'dataset': 'cifar',
        'data': 'data/cifar10',
        'arch': 'resnet50',
        'adv_train': 0,
        'adv_eval': 1,
        'constraint': '2',
        'eps': e,
        'attack_steps': 20,
        'attack_lr': 2.5 * e / 20,
        'fast_attack_lr': 1.5 * e
    }
    train_args = Parameters(eval_kwargs)

    # Fill whatever parameters are missing from the defaults
    train_args = setup_args(train_args)
    print(train_args)
    store = setup_store_with_metadata(train_args)

    final_model = main(train_args, store=store)

