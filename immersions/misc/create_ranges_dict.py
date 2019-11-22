from collections import OrderedDict
import pickle

ranges_dict =  OrderedDict([("scalogram", (88, 264)),  # 176
("scalogram_block_0_main_conv_1", (44, 132)),
("scalogram_block_0_main_conv_2", (44, 132)),
("scalogram_block_1_main_conv_1", (44, 132)),
("scalogram_block_1_main_conv_2", (44, 132)),
("scalogram_block_2_main_conv_1", (44, 132)),
("scalogram_block_2_main_conv_2", (44, 132)),
("scalogram_block_3_main_conv_1", (44, 132)),
("scalogram_block_3_main_conv_2", (44, 132)),  # 88
("scalogram_block_4_main_conv_1", (22, 66)),
("scalogram_block_4_main_conv_2", (22, 66)),
("scalogram_block_5_main_conv_1", (22, 66)),
("scalogram_block_5_main_conv_2", (22, 66)),
("scalogram_block_6_main_conv_1", (22, 66)),
("scalogram_block_6_main_conv_2", (22, 66)),
("scalogram_block_7_main_conv_1", (22, 66)),
("scalogram_block_7_main_conv_2", (21, 65)),  # 44
("ar_block_0", (19, 63)),
("ar_block_1", (18, 62)),
("ar_block_2", (9, 31)),
("ar_block_3", (8, 30)),
("ar_block_4", (7, 29)),
("ar_block_5", (3, 14)),
("ar_block_6", (2, 13)),
("ar_block_7", (2, 13)),
("ar_block_8", (0, 11))])


with open('/Volumes/Elements/Projekte/Immersions/models/immersions_scalogram_resnet_house_smaller/immersions_scalogram_resnet_house_smaller_ranges.p', 'wb') as handle:
    pickle.dump(ranges_dict, handle)