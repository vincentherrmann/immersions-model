from collections import OrderedDict
import pickle

# shapes_path = '/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13/activation_shapes.pickle'
#
# with open(shapes_path, 'rb') as handle:
#     shapes = pickle.load(handle)

activation_shapes = OrderedDict([('scalogram', [2, 216, 176]),
             ('scalogram_block_0_main_conv_1', [8, 108, 88]),
             ('scalogram_block_0_main_conv_2', [8, 84, 88]),
             ('scalogram_block_1_main_conv_1', [16, 84, 88]),
             ('scalogram_block_1_main_conv_2', [16, 84, 88]),
             ('scalogram_block_2_main_conv_1', [32, 84, 88]),
             ('scalogram_block_2_main_conv_2', [32, 60, 88]),
             ('scalogram_block_3_main_conv_1', [64, 60, 88]),
             ('scalogram_block_3_main_conv_2', [64, 60, 88]),
             ('scalogram_block_4_main_conv_1', [128, 30, 44]),
             ('scalogram_block_4_main_conv_2', [128, 6, 44]),
             ('scalogram_block_5_main_conv_1', [256, 6, 44]),
             ('scalogram_block_5_main_conv_2', [256, 6, 44]),
             ('scalogram_block_6_main_conv_1', [512, 6, 44]),
             ('scalogram_block_6_main_conv_2', [512, 3, 44]),
             ('scalogram_block_7_main_conv_1', [512, 3, 44]),
             ('scalogram_block_7_main_conv_2', [512, 1, 44]),
             ('z_code', [512, 44]),
             ('ar_block_0', [512, 44]),
             ('ar_block_1', [512, 44]),
             ('ar_block_2', [512, 22]),
             ('ar_block_3', [512, 22]),
             ('ar_block_4', [256, 22]),
             ('ar_block_5', [256, 11]),
             ('ar_block_6', [256, 11]),
             ('ar_block_7', [256, 11]),
             ('ar_block_8', [256, 11]),
             ('c_code', [256]),
             ('prediction', [16, 512])])

with open('immersions_scalogram_resnet_house_smaller_activation_shapes.p', 'wb') as handle:
    pickle.dump(activation_shapes, handle)