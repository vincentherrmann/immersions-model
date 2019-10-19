from collections import OrderedDict
import pickle

ranges_dict =  OrderedDict([("scalogram", (16, 368)),
("scalogram_block_0_main_conv_1", (16, 360)),
("scalogram_block_0_main_conv_2", (8, 180)),
("scalogram_block_1_main_conv_1", (8, 180)),
("scalogram_block_1_main_conv_2", (8, 180)),
("scalogram_block_2_main_conv_1", (8, 180)),
("scalogram_block_2_main_conv_2", (4, 90)),
("scalogram_block_3_main_conv_1", (4, 90)),
("scalogram_block_3_main_conv_2", (4, 90)),
("scalogram_block_4_main_conv_1", (4, 90)),
("scalogram_block_4_main_conv_2", (2, 45)),
("scalogram_block_5_main_conv_1", (2, 45)),
("scalogram_block_5_main_conv_2", (2, 45)),
("scalogram_block_6_main_conv_1", (2, 45)),
("scalogram_block_6_main_conv_2", (2, 45)),
("scalogram_block_7_main_conv_1", (1, 44)),
("scalogram_block_7_main_conv_2", (0, 43))])


with open('immersions_scalogram_resnet_ranges.p', 'wb') as handle:
    pickle.dump(ranges_dict, handle)