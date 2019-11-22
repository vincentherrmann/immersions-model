import torch
import pickle
from immersions.cpc_system import ContrastivePredictiveSystem
from immersions.cpc_system_maestro import ContrastivePredictiveSystemMaestro
from immersions.input_optimization.activation_utilities import ActivationStatistics
from pytorch_lightning.trainer.trainer_io import load_hparams_from_tags_csv




weights_path = '/home/idivinci3005/experiments/checkpoints/immersions_scalogram_resnet_maestro_smaller/0/_ckpt_epoch_3.ckpt'
tags_path = '/home/idivinci3005/experiments/logs/immersions_scalogram_resnet_maestro_smaller/version_0/meta_tags.csv'
ranges_path = '/home/idivinci3005/pycharm_immersions/immersions/misc/immersions_scalogram_resnet_house_smaller_ranges.p'

# weights_path = '/home/idivinci3005/experiments/checkpoints/immersions_scalogram_resnet_house/0/_ckpt_epoch_8.ckpt'
# tags_path = '/home/idivinci3005/experiments/logs/immersions_scalogram_resnet_house/version_0/meta_tags.csv'
# ranges_path = '/home/idivinci3005/pycharm_immersions/immersions/misc/immersions_scalogram_resnet_house_ranges.p'

hparams = load_hparams_from_tags_csv(tags_path)
hparams.visible_steps = 76
hparams.__setattr__('on_gpu', torch.cuda.is_available())

# load on CPU only to avoid OOM issues
# then its up to user to put back on GPUs
checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

# load the state_dict on the model automatically
#model = ContrastivePredictiveSystemMaestro(hparams)
model = ContrastivePredictiveSystemMaestro(hparams)
model.load_state_dict(checkpoint['state_dict'], strict=False)

# give model a chance to load something
model.on_load_checkpoint(checkpoint)

model.cuda()

model.freeze()
model.eval()

with open(ranges_path, 'rb') as handle:
    activation_ranges = pickle.load(handle)

result_dict = model.calc_silence_statistics(activation_ranges=activation_ranges, num_batches=10, device='cuda:0')

with open('immersions_scalogram_resnet_maestro_smaller_silence_statistics.p', 'wb') as handle:
    pickle.dump(result_dict, handle)

for key, value in result_dict['element_mean'].items():
    print(key)
    print(list(value.shape))


pass



