import torch
from fastai.basic_train import Learner, LearnerCallback, Callback, add_metrics
from fastai.callback import annealing_cos
from fastai.callbacks import SaveModelCallback
from fastai.callbacks.general_sched import TrainingPhase, GeneralScheduler
from losses_and_metrics import group_mean_log_mae, reshape_targs, contribs_rmse_loss
import constants as C
import pdb
import numpy as np
from visdom import Visdom

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom(port=5919)
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y, env=None, x_label='Epochs'):
        if env is not None:
            print_env = env
        else:
            print_env = self.env
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel=x_label,
                ylabel=var_name))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name], name=split_name, update='append')

plotter = VisdomLinePlotter(env_name='fangjia')

class GradientClipping(LearnerCallback):
    "Gradient clipping during training after 'start_it' number of steps."
    def __init__(self, learn:Learner, clip:float = 0., start_it:int = 100):
        super().__init__(learn)
        self.clip, self.start_it = clip, start_it

    def on_backward_end(self, iteration, **kwargs):
        "Clip the gradient before the optimizer step."
        if self.clip and (iteration > self.start_it):
            torch.nn.utils.clip_grad_norm_(
                self.learn.model.parameters(), self.clip)

class GroupMeanLogMAE(Callback):
    """Callback to report the group mean log MAE during training."""
    _order = -20 # Needs to run before the recorder
    metrics_list_train, metrics_list_valid = [], []
    sc_types_valid, output_valid, target_valid = [], [], []
    def __init__(self, learn, **kwargs):
        self.learn = learn

    def on_train_begin(self, **kwargs):
        metric_names = ['group_mean_log_mae']
        self.learn.recorder.add_metric_names(metric_names)

    def on_epoch_begin(self, **kwargs):
        self.sc_types_train, self.output_train, self.target_train = [], [], []
        self.sc_types_valid, self.output_valid, self.target_valid = [], [], []
    
#    def on_batch_begin(self, **kwargs):
#        self.sc_types_valid, self.output_valid, self.target_valid = [], [], []

    def on_batch_end(self, last_target, last_output, last_input, train, **kwargs):
        sc_types = last_input[-1].view(-1)
        mask = sc_types != C.BATCH_PAD_VAL
        if train:
            self.sc_types_train.append(sc_types[mask])
            self.output_train.append(last_output[:,-1])
            self.target_train.append(reshape_targs(last_target)[:,-1])
        else:
            #pdb.set_trace()
            self.sc_types_valid.append(sc_types[mask])
            self.output_valid.append(last_output[:,-1])
            self.target_valid.append(reshape_targs(last_target)[:,-1])
#            torch.save(self.sc_types_valid[-1], '/home/nesa/fangjia/kaggle-champs-master--0/sc_types_valid.pt')
#            torch.save(self.output_valid[-1], '/home/nesa/fangjia/kaggle-champs-master--0/output_valid.pt')
#            torch.save(self.target_valid[-1], '/home/nesa/fangjia/kaggle-champs-master--0/target_valid.pt')

    def on_epoch_end(self, epoch, last_metrics, **kwargs):
        if (len(self.sc_types_train) > 0) and (len(self.output_train) > 0):
            sc_types_train = torch.cat(self.sc_types_train)
            preds_train = torch.cat(self.output_train)
            target_train = torch.cat(self.target_train)
            metrics_train = [group_mean_log_mae(preds_train, target_train, sc_types_train, C.SC_MEAN, C.SC_STD)] 
            plotter.plot('MPNN_Transformer', 'train', epoch, metrics_train[0])
            self.metrics_list_train.append(metrics_train[0])
            #torch.save(self.metrics_list_train,'/home/nesa/fangjia/kaggle-champs-master--0/metrics_feat_attn_train.pt')           
            #torch.save(self.metrics_list_train,'/home/nesa/fangjia/kaggle-champs-master--0/metrics_train.pt')
            #torch.save(self.metrics_list_train,'/home/nesa/fangjia/kaggle-champs-master--0/metrics_attn_train.pt')
            torch.save(self.metrics_list_train,'/home/nesa/fangjia/kaggle-champs-master--0/metrics_feat_train.pt')
            
            sc_types_valid = torch.cat(self.sc_types_valid)
            print(sc_types_valid.shape)
            preds_valid = torch.cat(self.output_valid)
            target_valid = torch.cat(self.target_valid)     
            metrics_valid = [group_mean_log_mae(preds_valid, target_valid, sc_types_valid, C.SC_MEAN, C.SC_STD)]
            plotter.plot('MPNN_Transformer', 'valid', epoch, metrics_valid[0])
            self.metrics_list_valid.append(metrics_valid[0])
            #torch.save(self.metrics_list_valid,'/home/nesa/fangjia/kaggle-champs-master--0/metrics_feat_attn_valid.pt')
            #torch.save(self.metrics_list_valid,'/home/nesa/fangjia/kaggle-champs-master--0/metrics_valid.pt')
            #torch.save(self.metrics_list_valid,'/home/nesa/fangjia/kaggle-champs-master--0/metrics_attn_valid.pt')
            torch.save(self.metrics_list_valid,'/home/nesa/fangjia/kaggle-champs-master--0/metrics_feat_valid.pt')
            #return add_metrics(last_metrics, metrics_train)
            return add_metrics(last_metrics, metrics_valid)

# Fastai's automatic loading was causing CUDA memory errors during snapshot
# ensembling. The function below is a workaround.
def save_model_cb_jump_to_epoch_adj(cb, epoch:int)->None:
    """Overwrites standard jump_to_epoch for the SaveModelCallback."""
    print(f'Model {cb.name}_{epoch-1} not loaded.')
#SaveModelCallback.jump_to_epoch = save_model_cb_jump_to_epoch_adj
