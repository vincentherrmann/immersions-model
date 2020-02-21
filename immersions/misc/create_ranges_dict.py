from collections import OrderedDict
import pickle


ranges_dict = OrderedDict([("scalogram", 172),
("scalogram_block_0_main_conv_1", 86),
("scalogram_block_0_main_conv_2", 86),
("scalogram_block_1_main_conv_1", 86),
("scalogram_block_1_main_conv_2", 86),
("scalogram_block_2_main_conv_1", 86),
("scalogram_block_2_main_conv_2", 86),
("scalogram_block_3_main_conv_1", 86),
("scalogram_block_3_main_conv_2", 86),
("scalogram_block_4_main_conv_1", 43),
("scalogram_block_4_main_conv_2", 43),
("scalogram_block_5_main_conv_1", 43),
("scalogram_block_5_main_conv_2", 43),
("scalogram_block_6_main_conv_1", 43),
("scalogram_block_6_main_conv_2", 43),
("scalogram_block_7_main_conv_1", 43),
("scalogram_block_7_main_conv_2", 43),
("ar_block_0", 43),
("ar_block_1", 43),
("ar_block_2", 22),
("ar_block_3", 22),
("ar_block_4", 22),
("ar_block_5", 11),
("ar_block_6", 11),
("ar_block_7", 11),
("ar_block_8", 11)])

with open('/Volumes/Elements/Projekte/Immersions/models/immersions_maestro_new/activation_ranges.p', 'wb') as handle:
    pickle.dump(ranges_dict, handle)