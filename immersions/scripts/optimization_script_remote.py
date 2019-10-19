import time
import torch
from immersions_control_app.streaming import SocketDataExchangeServer
from immersions_control_app.control_utilities import activation_selection_dict
from immersions.input_optimization.optimization import Optimization, default_control_dict

# viz_control_server = SocketDataExchangeServer(#port=2222,
#                                               port=8765,
#                                               host='127.0.0.1',
#                                               stream_automatically=True)
#
# if not viz_control_server.check_receive(b'abc', timeout=None):
#     raise Exception("No connection established")
# else:
#     print("opt client online")
# viz_control_server.set_new_data(b'abc')
viz_control_server = None

start_control_dict = {
        'lr': -5.,
        'activation_loss': 0.,
        'noise_loss': 0.,
        'high_freq_loss': 0.,
        'time_masking': 0.,
        'pitch_masking': 0.,
        'eq_bands': [],
        'time_jitter': 0.,
        'batch_size': 4,
        'activation_selection': activation_selection_dict,
        'mix_original': 0.
    }

weights_path = '/home/idivinci3005/experiments/checkpoints/_ckpt_epoch_20.ckpt'
tags_path = '/home/idivinci3005/experiments/logs/immersions_scalogram_resnet/version_7/meta_tags.csv'
model_shapes_path = '/home/idivinci3005/experiments/immersions_files/immersions_scalogram_resnet/immersions_scalogram_resnet_activation_shapes.p'
ranges_path = '/home/idivinci3005/experiments/immersions_files/immersions_scalogram_resnet/immersions_scalogram_resnet_ranges.p'
noise_statistics_path = '/home/idivinci3005/experiments/immersions_files/immersions_scalogram_resnet/immersions_scalogram_resnet_silence_statistics.p'
data_statistics_path = '/home/idivinci3005/experiments/immersions_files/immersions_scalogram_resnet/immersions_scalogram_resnet_data_statistics.p'
soundclips_path = '/home/idivinci3005/souncdlip_44khz'

torch.autograd.set_detect_anomaly(False)
optimization = Optimization(weights_path=weights_path,
                            tags_path=tags_path,
                            model_shapes_path=model_shapes_path,
                            ranges_path=ranges_path,
                            noise_statistics_path=noise_statistics_path,
                            data_statistics_path=data_statistics_path,
                            soundclips_path=soundclips_path,
                            communicator=viz_control_server)

optimization.control_dict = start_control_dict
optimization.run()

time.sleep(30)