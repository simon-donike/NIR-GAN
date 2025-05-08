from omegaconf import ListConfig
from omegaconf import OmegaConf




def dataset_selector(config):
    dataset_type = config.Data.dataset_type

    # if omegaconf list, turn into list
    if isinstance(dataset_type, ListConfig):
        dataset_type = list(dataset_type)

    # in case is list and only 1, extrct dataset type
    if type(dataset_type) == list and len(dataset_type) == 1:
        dataset_type = str(dataset_type[0])
    
    # if we now have a string, load according dataset
    if type(dataset_type) == str:
        if dataset_type == "S2NAIP":
            from data.SEN2NAIP_dataset import S2NAIP_dm
            return S2NAIP_dm(config)
        elif dataset_type == "S2_rand":
            from data.S2_dataset import S2_datamodule
            return S2_datamodule(config)
        elif dataset_type == "worldstrat":
            from data.worldstrat import worldstrat_datamodule
            return worldstrat_datamodule(config)
        elif dataset_type == "S2_75k":
            from data.s2_75k_dataset import S2_75k_datamodule
            return S2_75k_datamodule(config)
        elif dataset_type == "S2_100k":
            from data.s100k_dataset import S2_100k_datamodule
            return S2_100k_datamodule(config)
        elif dataset_type == "L8_15k":
            from data.l8_15k_dataset import L8_15k_datamodule
            return L8_15k_datamodule(config)
        else:
            raise NotImplementedError(f"Dataset Type {dataset_type} not implemented")

    # if we have a list, load combined dataset by appending datasets
    elif type(dataset_type) == list:
        from data.combined_dataset_customizable import CombinedDataModule
        return CombinedDataModule(config, dataset_type)    
    
    # if we have neither, raise error
    else:
        raise ValueError("Invalid Dataset Type, must be string or list.")
    



if __name__=="__main__":
    import numpy as np
    from tqdm import tqdm
    from matplotlib import pyplot as plt

    def plot_mean_hist(dm,name,filename):


        mean_rgb = []
        mean_nir = []

        for i in tqdm(dm.train_dataloader()):
            mean_rgb.append(i["rgb"].mean().item())
            mean_nir.append(i["nir"].mean().item())

        # Compute histogram data
        bins = 200
        hist_rgb, bin_edges_rgb = np.histogram(mean_rgb, bins=bins, density=True)
        hist_nir, bin_edges_nir = np.histogram(mean_nir, bins=bins, density=True)

        # Compute bin centers
        bin_centers_rgb = (bin_edges_rgb[:-1] + bin_edges_rgb[1:]) / 2
        bin_centers_nir = (bin_edges_nir[:-1] + bin_edges_nir[1:]) / 2

        # Plot histograms as lines
        import matplotlib.pyplot as plt

        plt.plot(bin_centers_rgb, hist_rgb, label="RGB", linestyle='-', marker='', color='blue')
        plt.plot(bin_centers_nir, hist_nir, label="NIR", linestyle='-', marker='', color='red')

        # Add mean lines
        plt.axvline(np.mean(mean_rgb), color='blue', linestyle='dashed', label="RGB Mean")
        plt.axvline(np.mean(mean_nir), color='red', linestyle='dashed', label="NIR Mean")

        # Labels and legend
        plt.xlabel("Mean Value")
        plt.ylabel("Density")
        plt.legend()
        plt.title("Mean Distribution of RGB and NIR\n" + name)

        # Save plot
        plt.savefig("data/data_information/" + filename)
        plt.close()




    config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")
    dm = dataset_selector(config)
    

    name = "S2_100k"
    filename = "S2_100k.png"
    plot_mean_hist(dm,name=name,filename=filename)


