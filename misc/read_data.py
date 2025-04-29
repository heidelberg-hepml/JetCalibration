import uproot as up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
import os

path_data_folder = "/remote/gpu07/huetsch/JetCalibration/data_v2"

files = os.listdir(path_data_folder)
files = [file for file in files if file.endswith(".root")]


if False:
    for name in files:

        print("\n")
        print("Starting file: ", name)

        file = os.path.join(path_data_folder, name)
        file = up.open(file)
        filekeys = file.keys()
        filekeys.sort(key=lambda k: int(k.split(";")[-1]), reverse=True)
        latest_key = filekeys[0]

        tree = file[latest_key]
        keys = tree.keys()

        true_E = ak.to_numpy(tree["ak10_true_E"].array())/1000
        rec_E = ak.to_numpy(tree["ak10_E"].array())
        E_ratio = rec_E / true_E

        true_m = ak.to_numpy(tree["ak10_true_mass"].array())/1000
        rec_m = ak.to_numpy(tree["ak10_mass"].array())
        m_ratio = rec_m / true_m

        true_pt = ak.to_numpy(tree["ak10_true_pt"].array())
        rec_pt = ak.to_numpy(tree["ak10_pt"].array())

        true_pt_cut = true_pt > 100
        rec_pt_cut = rec_pt > 100
        true_mass_cut = true_m > 50
        rec_mass_cut = rec_m > 50
        full_cut = true_pt_cut & rec_pt_cut & true_mass_cut & rec_mass_cut

        keys_to_keep = [
            'ak10_E',
            'ak10_mass',
            'ak10_rap',
            'ak10_groomMRatio',
            'ak10_Width',
            'ak10_Split12',
            'ak10_Split23',
            'ak10_C2',
            'ak10_D2',
            'ak10_Tau21',
            'ak10_Tau32',
            'ak10_Qw',
            'ak10_EMFracCaloBased',
            'ak10_EM3FracCaloBased',
            'ak10_Tile0FracCaloBased',
            'ak10_EffNClustsCaloBased',
            'ak10_NeutralEFrac',
            'ak10_ChargePTFrac',
            'ak10_ChargeMFrac',
            'averageMu',
            'NPV',    
        ]

        inputs = np.array([ak.to_numpy(tree[key].array()) for key in keys_to_keep])
        full_data = np.concatenate([E_ratio[:, None], m_ratio[:, None], inputs.T], axis=1)
        print("Shape of full_data: ", full_data.shape)

        full_data = full_data[full_cut]
        print("Shape of full_data after cut: ", full_data.shape)

        # apply a log transformation to the target ratios
        full_data[:, 0] = np.log10(full_data[:, 0])
        full_data[:, 1] = np.log10(full_data[:, 1])

        nan_mask = np.isnan(full_data).any(1)
        full_data = full_data[~nan_mask]
        print("Shape of full_data after nan mask: ", full_data.shape)

        np.save(os.path.join(path_data_folder, name.replace(".root", ".npy")), full_data)


if False:
    files = os.listdir(path_data_folder)
    files = [file for file in files if file.endswith(".npy")]

    data = []
    for file in files:
        data.append(np.load(os.path.join(path_data_folder, file)))

    data = np.concatenate(data, axis=0)
    print(data.shape)

    np.save(os.path.join(path_data_folder, "full_data.npy"), data)
    full_data_mean, full_data_std = np.mean(data, axis=0), np.std(data, axis=0)
    np.savez(os.path.join(path_data_folder, "full_data_mean_std.npz"), mean=full_data_mean, std=full_data_std)


path = os.path.join(path_data_folder, "full_data.npy")
data = np.load(path)

#print(data.shape)

#nan_mask = np.isnan(data).any(axis=1)
#inf_mask = np.isinf(data).any(axis=1)
#data = data[~nan_mask & ~inf_mask]

#print(data.shape)

#print(np.isnan(data).any())
#print(np.isinf(data).any())
#np.save(path, data)



log_dims = [0, 1, 5, 6, 8, 11, 19, 20]
for dim in log_dims:
    data[:, dim+2] = np.log10(data[:, dim+2]+1)

mean, std = np.mean(data, axis=0), np.std(data, axis=0)
data = (data - mean) / std

print(np.isnan(mean))
print(np.isnan(std))
print(np.isinf(mean))
print(np.isinf(std))

print(np.isnan(data).any())
print(np.isinf(data).any())

np.savez(os.path.join(path_data_folder, "full_data_mean_std.npz"), mean=mean, std=std)
np.save(os.path.join(path_data_folder, "full_data_preprocessed.npy"), data)
