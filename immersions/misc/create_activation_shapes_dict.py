from collections import OrderedDict
import pickle

# shapes_path = '/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13/activation_shapes.pickle'
#
# with open(shapes_path, 'rb') as handle:
#     shapes = pickle.load(handle)

activation_shapes = OrderedDict([('scalogram', [2, 216, 172]),
             ('scalogram_block_0_main_conv_1', [8, 107, 86]),
             ('scalogram_block_0_main_conv_2', [8, 83, 86]),
             ('scalogram_block_1_main_conv_1', [16, 81, 86]),
             ('scalogram_block_1_main_conv_2', [16, 79, 86]),
             ('scalogram_block_2_main_conv_1', [32, 77, 86]),
             ('scalogram_block_2_main_conv_2', [32, 63, 86]),
             ('scalogram_block_3_main_conv_1', [64, 61, 86]),
             ('scalogram_block_3_main_conv_2', [64, 59, 86]),
             ('scalogram_block_4_main_conv_1', [128, 29, 43]),
             ('scalogram_block_4_main_conv_2', [128, 15, 43]),
             ('scalogram_block_5_main_conv_1', [256, 13, 43]),
             ('scalogram_block_5_main_conv_2', [256, 11, 43]),
             ('scalogram_block_6_main_conv_1', [512, 9, 43]),
             ('scalogram_block_6_main_conv_2', [512, 5, 43]),
             ('scalogram_block_7_main_conv_1', [512, 3, 43]),
             ('scalogram_block_7_main_conv_2', [512, 1, 43]),
             ('z_code', [512, 1, 43]),
             ('ar_block_0', [512, 1, 43]),
             ('ar_block_1', [512, 1, 43]),
             ('ar_block_2', [512, 1, 22]),
             ('ar_block_3', [512, 1, 22]),
             ('ar_block_4', [256, 1, 22]),
             ('ar_block_5', [256, 1, 11]),
             ('ar_block_6', [256, 1, 11]),
             ('ar_block_7', [256, 1, 11]),
             ('ar_block_8', [256, 1, 11]),
             ('c_code', [256, 1, 11]),
             ('prediction', [16, 1, 512])])

model_path = "/Volumes/Elements/Projekte/Immersions/models/immersions_maestro_new"
with open(model_path + '/activation_shapes.p', 'wb') as handle:
    pickle.dump(activation_shapes, handle)