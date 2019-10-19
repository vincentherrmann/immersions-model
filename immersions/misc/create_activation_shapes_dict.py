from collections import OrderedDict
import pickle

shapes_path = '/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13/activation_shapes.pickle'

with open(shapes_path, 'rb') as handle:
    shapes = pickle.load(handle)

activation_shapes = OrderedDict([('scalogram', [2, 292, 352]),
             ('scalogram_block_0_main_conv_1', [32, 228, 344]),
             ('scalogram_block_0_main_conv_2', [32, 114, 172]),
             ('scalogram_block_1_main_conv_1', [32, 114, 172]),
             ('scalogram_block_1_main_conv_2', [32, 114, 172]),
             ('scalogram_block_2_main_conv_1', [64, 82, 172]),
             ('scalogram_block_2_main_conv_2', [64, 41, 86]),
             ('scalogram_block_3_main_conv_1', [64, 41, 86]),
             ('scalogram_block_3_main_conv_2', [64, 41, 86]),
             ('scalogram_block_4_main_conv_1', [128, 26, 86]),
             ('scalogram_block_4_main_conv_2', [128, 13, 43]),
             ('scalogram_block_5_main_conv_1', [128, 13, 43]),
             ('scalogram_block_5_main_conv_2', [128, 13, 43]),
             ('scalogram_block_6_main_conv_1', [256, 5, 43]),
             ('scalogram_block_6_main_conv_2', [256, 5, 43]),
             ('scalogram_block_7_main_conv_1', [512, 3, 43]),
             ('scalogram_block_7_main_conv_2', [512, 1, 43]),
             ('z_code', [512, 43]),
             ('ar_block_0', [512, 39]),
             ('ar_block_1', [512, 36]),
             ('ar_block_2', [512, 18]),
             ('ar_block_3', [512, 16]),
             ('ar_block_4', [256, 14]),
             ('ar_block_5', [256, 7]),
             ('ar_block_6', [256, 5]),
             ('ar_block_7', [256, 5]),
             ('ar_block_8', [256, 1]),
             ('c_code', [256]),
             ('prediction', [16, 512])])

with open('immersions_scalogram_resnet_activation_shapes.p', 'wb') as handle:
    pickle.dump(activation_shapes, handle)