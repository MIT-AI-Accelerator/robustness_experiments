import os

if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

import time
import torch
import torch as ch
from torchvision.utils import make_grid

from cox.utils import Parameters
import cox.store
import dill

from robustness import model_utils, datasets, train, defaults
from robustness.attacker import Attacker, AttackerModel
from robustness.train import eval_model, check_required_args, make_optimizer_and_schedule
from robustness.tools import helpers, constants
from robustness import attack_steps

STEPS = {
    'inf': attack_steps.LinfStep,
    '2': attack_steps.L2Step,
    'unconstrained': attack_steps.UnconstrainedStep,
    'fourier': attack_steps.FourierStep
}

consts = constants
has_attr = helpers.has_attr
AverageMeter = helpers.AverageMeter
calc_fadein_eps = helpers.calc_fadein_eps
ckpt_at_epoch = helpers.ckpt_at_epoch


class AttackerFGSM(ch.nn.Module):
    def __init__(self, model, dataset):
        """
        Initialize the Attacker

        Args:
            nn.Module model : the PyTorch model to attack
            Dataset dataset : dataset the model is trained on, only used to get mean and std for normalization
        """
        super(AttackerFGSM, self).__init__()
        self.normalize = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model

    def forward(self, x, target, *_, constraint, eps, step_size, iterations,
                random_start=False, random_restarts=False, do_tqdm=False,
                targeted=False, custom_loss=None, should_normalize=True,
                orig_input=None, use_best=True, return_image=True, est_grad=None):
        """
        Implementation of forward (finds adversarial examples). Note that
        this does **not** perform inference and should not be called
        directly; refer to :meth:`robustness.attacker.AttackerModel.forward`
        for the function you should actually be calling.

        Args:
            x, target (ch.tensor) : see :meth:`robustness.attacker.AttackerModel.forward`
            constraint
                ("2"|"inf"|"unconstrained"|"fourier"|:class:`~robustness.attack_steps.AttackerStep`)
                : threat model for adversarial attacks (:math:`\ell_2` ball,
                :math:`\ell_\infty` ball, :math:`[0, 1]^n`, Fourier basis, or
                custom AttackerStep subclass).
            eps (float) : radius for threat model.
            step_size (float) : step size for adversarial attacks.
            random_start (bool) : if True, start the attack with a random step.
            random_restarts (bool) : if True, do many random restarts and
                take the worst attack (in terms of loss) per input.
            targeted (bool) : if True (False), minimize (maximize) the loss.
            custom_loss (function|None) : if provided, used instead of the
                criterion as the loss to maximize/minimize during
                adversarial attack. The function should take in
                :samp:`model, x, target` and return a tuple of the form
                :samp:`loss, None`, where loss is a tensor of size N
                (per-element loss).
            should_normalize (bool) : If False, don't normalize the input
                (not recommended unless normalization is done in the
                custom_loss instead).
            orig_input (ch.tensor|None) : If not None, use this as the
                center of the perturbation set, rather than :samp:`x`.
            return_image (bool) : If True (default), then return the adversarial
                example as an image, otherwise return it in its parameterization
                (for example, the Fourier coefficients if 'constraint' is
                'fourier')
            est_grad (tuple|None) : If not None (default), then these are
                :samp:`(query_radius [R], num_queries [N])` to use for estimating the
                gradient instead of autograd. We use the spherical gradient
                estimator, shown below, along with antithetic sampling [#f1]_
                to reduce variance:
                :math:`\\nabla_x f(x) \\approx \\sum_{i=0}^N f(x + R\\cdot
                \\vec{\\delta_i})\\cdot \\vec{\\delta_i}`, where
                :math:`\delta_i` are randomly sampled from the unit ball.
        Returns:
            An adversarial example for x (i.e. within a feasible set
            determined by `eps` and `constraint`, but classified as:

            * `target` (if `targeted == True`)
            *  not `target` (if `targeted == False`)

        .. [#f1] This means that we actually draw :math:`N/2` random vectors
            from the unit ball, and then use :math:`\delta_{N/2+i} =
            -\delta_{i}`.
        """

        # Can provide a different input to make the feasible set around
        # instead of the initial point
        if orig_input is None: orig_input = x.detach()
        orig_input = orig_input.cuda()

        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if targeted else 1

        # Initialize step class and attacker criterion
        criterion = ch.nn.CrossEntropyLoss(reduction='none').cuda()
        step_class = STEPS[constraint] if isinstance(constraint, str) else constraint
        step = step_class(eps=eps, orig_input=orig_input, step_size=step_size)

        def calc_loss(inp, target):
            '''
            Calculates the loss of an input with respect to target labels
            Uses custom loss (if provided) otherwise the criterion
            '''
            if should_normalize:
                inp = self.normalize(inp)
            output = self.model(inp)
            if custom_loss:
                return custom_loss(self.model, inp, target)

            return criterion(output, target), output

        # Main function for making adversarial examples
        def get_adv_examples(x):
            # Random start (to escape certain types of gradient masking)
            if random_start:
                x = step.random_perturb(x)

            # Fast Adversaril with FGSM
            x = x.clone().detach()
            δ = step.random_perturb(ch.zeros_like(x)).requires_grad_(True)
            losses, _ = calc_loss(step.to_image(x + δ), target)
            assert losses.shape[0] == x.shape[0], \
                    'Shape of losses must match input!'

            loss = ch.mean(losses)

            if step.use_grad:
                if est_grad is None:
                    grad, = ch.autograd.grad(m * loss, [δ])
                else:
                    f = lambda _x, _y: m * calc_loss(step.to_image(_x), _y)[0]
                    grad = helpers.calc_est_grad(f, xd, target, *est_grad)
            else:
                grad = None

            with ch.no_grad():
                δ = step.step(δ, grad) # δ = δ + α ∇_δ (f(x + δ))
                x = step.project(x + δ)

            # Save computation (don't compute last loss) if not use_best
            ret = x.clone().detach()
            return step.to_image(ret) if return_image else ret

        # Random restarts: repeat the attack and find the worst-case
        # example for each input in the batch
        if random_restarts:
            to_ret = None

            orig_cpy = x.clone().detach()
            for _ in range(random_restarts):
                adv = get_adv_examples(orig_cpy)

                if to_ret is None:
                    to_ret = adv.detach()

                _, output = calc_loss(adv, target)
                corr, = helpers.accuracy(output, target, topk=(1,), exact=True)
                corr = corr.byte()
                misclass = ~corr
                to_ret[misclass] = adv[misclass]

            adv_ret = to_ret
        else:
            adv_ret = get_adv_examples(x)

        return adv_ret

class AttackerModelFGSM(AttackerModel):
    def __init__(self, model, dataset, attacker, attacker_val):
        super().__init__(model, dataset)
        self.attacker = attacker
        self.attacker_val = attacker_val

    def forward(self, inp, target=None, make_adv=False, with_latent=False,
                fake_relu=False, no_relu=False, with_image=True, **attacker_kwargs):

        if make_adv:
            assert target is not None
            prev_training = bool(self.training)
            self.eval()
            if prev_training:
                adv = self.attacker(inp, target, **attacker_kwargs)
                self.train()
            else:
                adv = self.attacker_val(inp, target, **attacker_kwargs)

            inp = adv

        if with_image:
            normalized_inp = self.normalizer(inp)

            if no_relu and (not with_latent):
                print("WARNING: 'no_relu' has no visible effect if 'with_latent is False.")
            if no_relu and fake_relu:
                raise ValueError("Options 'no_relu' and 'fake_relu' are exclusive")

            output = self.model(normalized_inp, with_latent=with_latent,
                                    fake_relu=fake_relu, no_relu=no_relu)
        else:
            output = None

        return (output, inp)

def make_and_restore_model(*_, arch, dataset, resume_path=None,
         parallel=True, pytorch_pretrained=False):
    """
    Makes a model and (optionally) restores it from a checkpoint.

    Args:
        arch (str|nn.Module): Model architecture identifier or otherwise a
            torch.nn.Module instance with the classifier
        dataset (Dataset class [see datasets.py])
        resume_path (str): optional path to checkpoint
        parallel (bool): if True, wrap the model in a DataParallel
            (default True, recommended)
        pytorch_pretrained (bool): if True, try to load a standard-trained
            checkpoint from the torchvision library (throw error if failed)
    Returns:
        A tuple consisting of the model (possibly loaded with checkpoint), and the checkpoint itself
    """
    classifier_model = dataset.get_model(arch, pytorch_pretrained) if \
                            isinstance(arch, str) else arch

    attacker_pgd = Attacker(classifier_model, dataset)
    attacker = AttackerFGSM(classifier_model, dataset)
    model = AttackerModelFGSM(classifier_model, dataset, attacker, attacker_pgd)

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path:
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = ch.load(resume_path, pickle_module=dill)

            # Makes us able to load models saved with legacy versions
            state_dict_path = 'model'
            if not ('model' in checkpoint):
                state_dict_path = 'state_dict'

            sd = checkpoint[state_dict_path]
            sd = {k[len('module.'):]:v for k,v in sd.items()}
            model.load_state_dict(sd)
            if parallel:
                model = ch.nn.DataParallel(model)
            model = model.cuda()

            print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            error_msg = "=> no checkpoint found at '{}'".format(resume_path)
            raise ValueError(error_msg)

    return model, checkpoint

def train_model(args, model, loaders, *, checkpoint=None,
                store=None, update_params=None):
    # Logging setup
    writer = store.tensorboard if store else None
    if store is not None:
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)
        store.add_table(consts.CKPTS_TABLE, consts.CKPTS_SCHEMA)

    # Reformat and read arguments
    check_required_args(args) # Argument sanity check
    args.eps = eval(str(args.eps)) if has_attr(args, 'eps') else None
    args.attack_lr = eval(str(args.attack_lr)) if has_attr(args, 'attack_lr') else None

    # Initial setup
    train_loader, val_loader = loaders
    opt, schedule = make_optimizer_and_schedule(args, model, checkpoint, update_params)

    best_prec1, start_epoch = (0, 0)
    if checkpoint:
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint[f"{'adv' if args.adv_train else 'nat'}_prec1"]

    # Put the model into parallel mode
    assert not hasattr(model, "module"), "model is already in DataParallel."
    model = ch.nn.DataParallel(model).cuda()

    # Timestamp for training start time
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        train_prec1, train_loss = _model_loop(args, 'train', train_loader,
                model, opt, epoch, args.adv_train, writer)
        last_epoch = (epoch == (args.epochs - 1))

        # evaluate on validation set
        sd_info = {
            'model':model.state_dict(),
            'optimizer':opt.state_dict(),
            'schedule':(schedule and schedule.state_dict()),
            'epoch': epoch+1
        }

        def save_checkpoint(filename):
            ckpt_save_path = os.path.join(args.out_dir if not store else \
                                          store.path, filename)
            ch.save(sd_info, ckpt_save_path, pickle_module=dill)

        save_its = args.save_ckpt_iters
        should_save_ckpt = (epoch % save_its == 0) and (save_its > 0)
        should_log = (epoch % args.log_iters == 0)

        if should_log or last_epoch or should_save_ckpt:
            # log + get best
            with ch.no_grad():
                prec1, nat_loss = _model_loop(args, 'val', val_loader, model,
                        None, epoch, False, writer)

            # loader, model, epoch, input_adv_exs
            should_adv_eval = args.adv_eval or args.adv_train
            adv_val = should_adv_eval and _model_loop(args, 'val', val_loader,
                    model, None, epoch, True, writer)
            adv_prec1, adv_loss = adv_val or (-1.0, -1.0)

            # remember best prec@1 and save checkpoint
            our_prec1 = adv_prec1 if args.adv_train else prec1
            is_best = our_prec1 > best_prec1
            best_prec1 = max(our_prec1, best_prec1)

            # log every checkpoint
            log_info = {
                'epoch':epoch + 1,
                'nat_prec1':prec1,
                'adv_prec1':adv_prec1,
                'nat_loss':nat_loss,
                'adv_loss':adv_loss,
                'train_prec1':train_prec1,
                'train_loss':train_loss,
                'time': time.time() - start_time
            }

            # Log info into the logs table
            if store: store[consts.LOGS_TABLE].append_row(log_info)
            # If we are at a saving epoch (or the last epoch), save a checkpoint
            if should_save_ckpt or last_epoch: save_checkpoint(ckpt_at_epoch(epoch))
            # If  store exists and this is the last epoch, save a checkpoint
            if last_epoch and store: store[consts.CKPTS_TABLE].append_row(sd_info)

            # Update the latest and best checkpoints (overrides old one)
            save_checkpoint(consts.CKPT_NAME_LATEST)
            if is_best: save_checkpoint(consts.CKPT_NAME_BEST)

        if schedule: schedule.step()
        if has_attr(args, 'epoch_hook'): args.epoch_hook(model, log_info)

    return model

def _model_loop(args, loop_type, loader, model, opt, epoch, adv, writer):
    if not loop_type in ['train', 'val']:
        err_msg = "loop_type ({0}) must be 'train' or 'val'".format(loop_type)
        raise ValueError(err_msg)
    is_train = (loop_type == 'train')

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    prec = 'NatPrec' if not adv else 'AdvPrec'
    loop_msg = 'Train' if loop_type == 'train' else 'Val'

    # switch to train/eval mode depending
    model = model.train() if is_train else model.eval()

    # If adv training (or evaling), set eps and random_restarts appropriately
    if adv:
        eps = calc_fadein_eps(epoch, args.eps_fadein_epochs, args.eps) \
                if is_train else args.eps
        random_restarts = 0 if is_train else args.random_restarts

    # Custom training criterion
    has_custom_train_loss = has_attr(args, 'custom_train_loss')
    train_criterion = args.custom_train_loss if has_custom_train_loss \
            else ch.nn.CrossEntropyLoss()

    has_custom_adv_loss = has_attr(args, 'custom_adv_loss')
    adv_criterion = args.custom_adv_loss if has_custom_adv_loss else None

    attack_kwargs = {}
    if adv:
        if loop_type == 'train':
            attack_kwargs = {
                'constraint': args.constraint,
                'eps': eps,
                'iterations': 1,
                'step_size': args.fast_attack_lr,
                'random_start': args.random_start,
                'custom_loss': adv_criterion,
                'random_restarts': random_restarts,
            }

        elif loop_type == 'val':
            attack_kwargs = {
                'constraint': args.constraint,
                'eps': eps,
                'step_size': args.attack_lr,
                'iterations': args.attack_steps,
                'random_start': args.random_start,
                'custom_loss': adv_criterion,
                'random_restarts': random_restarts,
                'use_best': bool(args.use_best)
            }


    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
       # measure data loading time
        target = target.cuda(non_blocking=True)
        output, final_inp = model(inp,
                                  target=target,
                                  make_adv=adv,
                                  **attack_kwargs)
        loss = train_criterion(output, target)

        if len(loss.shape) > 0: loss = loss.mean()

        model_logits = output[0] if (type(output) is tuple) else output

        # measure accuracy and record loss
        top1_acc = float('nan')
        top5_acc = float('nan')
        try:
            maxk = min(5, model_logits.shape[-1])
            prec1, prec5 = helpers.accuracy(model_logits, target, topk=(1, maxk))

            losses.update(loss.item(), inp.size(0))
            top1.update(prec1[0], inp.size(0))
            top5.update(prec5[0], inp.size(0))

            top1_acc = top1.avg
            top5_acc = top5.avg
        except:
            pass

        reg_term = 0.0
        if has_attr(args, "regularizer"):
            reg_term =  args.regularizer(model, inp, target)
        loss = loss + reg_term

        # compute gradient and do SGD step
        if is_train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        elif adv and i == 0 and writer:
            # add some examples to the tensorboard
            nat_grid = make_grid(inp[:15, ...])
            adv_grid = make_grid(final_inp[:15, ...])
            writer.add_image('Nat input', nat_grid, epoch)
            writer.add_image('Adv input', adv_grid, epoch)

        # ITERATOR
        desc = ('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
                '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | '
                'Reg term: {reg} ||'.format( epoch, prec, loop_msg,
                loss=losses, top1_acc=top1_acc, top5_acc=top5_acc, reg=reg_term))

        # USER-DEFINED HOOK
        if has_attr(args, 'iteration_hook'):
            args.iteration_hook(model, i, loop_type, inp, target)

        iterator.set_description(desc)
        iterator.refresh()

    if writer is not None:
        prec_type = 'adv' if adv else 'nat'
        descs = ['loss', 'top1', 'top5']
        vals = [losses, top1, top5]
        for d, v in zip(descs, vals):
            writer.add_scalar('_'.join([prec_type, loop_type, d]), v.avg,
                              epoch)

    return top1.avg, losses.avg

def main(args, store=None):
    '''Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    '''
    # MAKE DATASET AND LOADERS
    data_path = os.path.expandvars(args.data)
    dataset = datasets.DATASETS[args.dataset](data_path)

    train_loader, val_loader = dataset.make_loaders(args.workers,
                    args.batch_size, data_aug=bool(args.data_aug))

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    loaders = (train_loader, val_loader)

    # MAKE MODEL
    model, checkpoint = make_and_restore_model(arch=args.arch,
            dataset=dataset, resume_path=args.resume)
    if 'module' in dir(model): model = model.module

    print(args)
    if args.eval_only:
        return eval_model(args, model, val_loader, store=store)

    model = train_model(args, model, loaders, store=store)
    return model

if __name__ == '__main__':
    from robustness.main import setup_args, setup_store_with_metadata

    train_kwargs = {
        'out_dir': "test",
        'exp_name': "test",
        'dataset': 'cifar',
        'data': '/home/justin_goodwin/data/cifar10',
        'arch': 'resnet50',
        'adv_train': 1,
        'constraint': '2',
        'eps': 0.25,
        'attack_steps': 20,
        'attack_lr': 2.5 * 0.25 / 20,
        'fast_attack_lr': 1.5 * 0.25
    }
    train_args = Parameters(train_kwargs)

    # Fill whatever parameters are missing from the defaults
    train_args = setup_args(train_args)
    print(train_args)
    store = setup_store_with_metadata(train_args)

    final_model = main(train_args, store=store)

