import pickle

with open('immersions_scalogram_resnet_silence_statistics.p', 'rb') as handle:
    silence_statistics = pickle.load(handle)

with open('immersions_scalogram_resnet_data_statistics.p', 'rb') as handle:
    data_statistics = pickle.load(handle)

pass