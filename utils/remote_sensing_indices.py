import torch


class RemoteSensingIndices():
    
    def __init__(self,mode="loss",criterion="l1"):
        
        # define mode of operations:
        # - loss: compute loss
        # - index: return index as array
        assert mode in ["loss","index"], f"Mode '{mode}' not implemented. 'loss', 'index' are supported."
        self.mode = mode
        
        # set criterion for loss
        if criterion=="l1":
            self.criterion = torch.nn.functional.l1_loss
        elif criterion=="l2":
            self.criterion = torch.nn.functional.mse_loss
        else:
            raise NotImplementedError(f"Criterion '{criterion}' not implemented. 'l1' or 'l2' are supported.")
        

    def get_and_weight_losses(self,rgb,nir,nir_pred,loss_config=None,mode="loss"):
        """
        Calcualates all metrics and weights them according to the loss_config.

        Args:
            rgb (3-band tensor): rgb image
            nir (1-band tensor): nir target
            nir_pred (1-band tensor): NIR prediction
            loss_config (dict): Dict of loss weights. Defaults to None.
            mode (str): Mode of operation. 'loss' or 'logging_dict'. Defaults to "loss".
        Returns:
            float: Total weighted loss
        """
        
        if loss_config==None: # if no config, define default
            loss_config = {"lambda_ndvi": 0.333,
                "lambda_ndwi": 0.333,
                "lambda_evi": 0.333,
                "lambda_savi": 0.0,
                "lambda_msavi": 0.0,
                "lambda_gndvi": 0.0}
            
        loss_fns = {
                "lambda_ndvi": ("indices_loss/ndvi_error", self.ndvi_calculation),
                "lambda_ndwi": ("indices_loss/ndwi_error", self.ndwi_calculation),
                "lambda_gndvi": ("indices_loss/gndvi_error", self.gndvi_calculation),
                "lambda_savi": ("indices_loss/savi_error", self.savi_calculation),
                "lambda_msavi": ("indices_loss/msavi_error", self.msavi_calculation),
                "lambda_evi": ("indices_loss/evi_error", self.evi_calculation),
            }

            
        if mode == "loss":
            total_loss = 0.0
            for loss_key, (_, loss_fn) in loss_fns.items():
                weight = loss_config.get(loss_key, 0.0)
                if weight > 0.0:
                    loss_value = loss_fn(rgb, nir, nir_pred)
                    total_loss += weight * loss_value
            return total_loss

        elif mode == "logging_dict":
            log_dict = {}
            for loss_key, (log_name, loss_fn) in loss_fns.items():
                loss_value = loss_fn(rgb, nir, nir_pred)
                log_dict[log_name] = loss_value
            return log_dict
        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented. 'loss' or 'logging_dict' are supported.")
        

    def prepare_tensor_for_loss(self,rgb,nir,nir_pred):
        if len(rgb.shape) == 3:
            rgb = rgb.unsqueeze(0)
        if len(nir.shape) == 3:
            nir = nir.unsqueeze(0)
        if len(nir_pred.shape) == 3:
            nir_pred = nir_pred.unsqueeze(0)
        return (rgb,nir,nir_pred)


    def ndvi_calculation(self,rgb,nir,nir_pred):
        """
        Calculate the NDVI loss between two images.

        Parameters:
        rgb (torch.Tensor): The RGB image.
        nir (torch.Tensor): The NIR image.
        nir_pred (torch.Tensor): The predicted NIR image.

        Returns:
        float: The NDVI loss between the NIR and predicted NIR images.
        
        Formula:
        (NIR - Red) / (NIR + Red)
        
        Purpose:
        Used to identify and monitor water bodies, helpful in determining moisture content and water stress in vegetation.
        """
        rgb,nir,nir_pred= self.prepare_tensor_for_loss(rgb,nir,nir_pred)
        
        # if loss, make sure division isnt by 0
        epsilon = 1e-6 if self.mode=="loss" else 0
        
        # Compute NDVI
        red = rgb[:,0:1,:,:]
        ndvi = (nir - red) / (nir + red + epsilon)
        ndvi_pred = (nir_pred - red) / (nir_pred + red + epsilon)

        if self.mode=="loss":
            #ndvi,ndvi_pred = (ndvi+1)/2,(ndvi_pred+1)/2 # add 1 to bring range from 0..2
            loss = self.criterion(ndvi, ndvi_pred)
            return loss
        elif self.mode=="index":
            return (ndvi,ndvi_pred)
        else:
            raise NotImplementedError(f"Mode '{self.mode}' not implemented. 'loss' or 'index' are supported.")



    def ndwi_calculation(self,rgb,nir,nir_pred):
        """
        Calculate the NDWI loss between two images.

        Parameters:
        rgb (torch.Tensor): The RGB image.
        nir (torch.Tensor): The NIR image.
        nir_pred (torch.Tensor): The predicted NIR image.

        Returns:
        float: The NDWI loss between the NIR and predicted NIR images.
        
        Formula:
        https://en.wikipedia.org/wiki/Normalized_difference_water_index
        (NIR - Green) / (NIR + Green)
        
        Purpose:
        Used to identify and monitor water bodies, helpful in determining moisture content and water stress in vegetation.
        """
        rgb,nir,nir_pred= self.prepare_tensor_for_loss(rgb,nir,nir_pred)
        
        # if loss, make sure division isnt by 0
        epsilon = 1e-6 if self.mode=="loss" else 0
        
        # Compute NDWI
        green = rgb[:,1:2,:,:]
        ndwi = (nir - green) / (nir + green + epsilon)
        ndwi_pred = (nir_pred - green) / (nir_pred + green + epsilon)

        if self.mode=="loss":
            #ndwi,ndwi_pred = (ndwi+1)/2,(ndwi_pred+1)/2 # add 1 to bring range from 0..2
            loss = self.criterion(ndwi, ndwi_pred)
            return loss
        elif self.mode=="index":
            return (ndwi,ndwi_pred)
        else:
            raise NotImplementedError(f"Mode '{self.mode}' not implemented. 'loss' or 'index' are supported.")


    def gndvi_calculation(self,rgb, nir, nir_pred):
        """
        Calculate the GNDVI loss between two images.

        Parameters:
        rgb (torch.Tensor): The RGB image. Assumes the channel order is RGB.
        nir (torch.Tensor): The NIR image.
        nir_pred (torch.Tensor): The predicted NIR image.

        Returns:
        float: The GNDVI loss between the NIR and predicted NIR images.
        
        Formula:
        https://www.auravant.com/en/help-en/imagery-and-layers/3636624-what-is-the-gndvi/
        GNDVI = (NIR-GREEN) /(NDVI+GREEN)        
        Purpose:
        Similar to NDVI but uses the green band instead of the red, enhancing sensitivity to chlorophyll content, thus providing a better indicator of vegetation health.
        """
        # Extract Green channel from RGB image
        green = rgb[:, 1:2, :, :]  # Assuming rgb is in BCHW format and channel order is RGB
        red = rgb[:, 0:1, :, :]
        
        # Compute NDVI
        ndvi = (nir - red) / (nir + red)
        ndvi_pred = (nir_pred - red) / (nir_pred + red)
        
        # Compute GNDVI
        gndvi = (nir - green) / (ndvi + green)
        gndvi_pred = (nir_pred - green) / (ndvi_pred + green)

        if self.mode=="loss":
            #gndvi,gndvi_pred = (gndvi+1)/2,(gndvi_pred+1)/2 # add 1 to bring range from 0..2
            loss = self.criterion(gndvi, gndvi_pred)
            return loss
        elif self.mode=="index":
            return (gndvi,gndvi_pred)
        else:
            raise NotImplementedError(f"Mode '{self.mode}' not implemented. 'loss' or 'index' are supported.")


    def savi_calculation(self,rgb, nir, nir_pred):
        """
        Calculate the SAVI loss between two images.

        Parameters:
        rgb (torch.Tensor): The RGB image. Assumes the channel order is RGB.
        nir (torch.Tensor): The NIR image.
        nir_pred (torch.Tensor): The predicted NIR image.

        Returns:
        float: The SAVI loss between the NIR and predicted NIR images.
        
        Formula:
        https://en.wikipedia.org/wiki/Soil-adjusted_vegetation_index
        SAVI = (1+L)(NIR-RED)/(NIR+RED+L)
        
        Purpose:
        Designed to minimize the influence of soil brightness, particularly useful when vegetation cover is low.
        L is a correction factor that varies (commonly 0.5).    
        """
        # Extract Red channel from RGB image
        red = rgb[:, 0:1, :, :]
        
        # Compute SAVI
        savi = (1 + 0.5) * (nir - red) / (nir + red + 0.5)
        savi_pred = (1 + 0.5) * (nir_pred - red) / (nir_pred + red + 0.5)

        if self.mode=="loss":
            #save,savi_pred = (savi+1)/2,(savi_pred+1)/2 # add 1 to bring range from 0..2
            loss = self.criterion(savi, savi_pred)
            return loss
        elif self.mode=="index":
            return (savi,savi_pred)
        else:
            raise NotImplementedError(f"Mode '{self.mode}' not implemented. 'loss' or 'index' are supported.")

        
    def msavi_calculation(self,rgb,nir,nir_pred):
        """
        Calculate the MSAVI loss between two images.

        Parameters:
        rgb (torch.Tensor): The RGB image. Assumes the channel order is RGB.
        nir (torch.Tensor): The NIR image.
        nir_pred (torch.Tensor): The predicted NIR image.

        Returns:
        float: The MSAVI loss between the NIR and predicted NIR images.
        
        Formula:
        https://www.usgs.gov/landsat-missions/landsat-modified-soil-adjusted-vegetation-index
        (2 * NIR + 1 – sqrt ((2 * NIR + 1)**2 – 8 * (NIR - R))) / 2
        
        Purpose:
        A modification of MSAVI that improves sensitivity in areas with sparse vegetation without
        requiring a soil brightness correction factor.  
        """
        rgb,nir,nir_pred= self.prepare_tensor_for_loss(rgb,nir,nir_pred)
        # extract red band
        red = rgb[:,0:1,:,:]
        
        # Compute MSAVI
        msavi = (2 * nir + 1 - torch.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
        msavi_pred = (2 * nir_pred + 1 - torch.sqrt((2 * nir_pred + 1)**2 - 8 * (nir_pred - red))) / 2
        
        if self.mode=="loss":
            #msavi,msavi_pred = (msavi+1)/2,(msavi_pred+1)/2 # add 1 to bring range from 0..2
            loss = self.criterion(msavi, msavi_pred)
            return(loss)
        elif self.mode=="index":
            return (msavi,msavi_pred)
        else:
            raise NotImplementedError(f"Mode '{self.mode}' not implemented. 'loss' or 'index' are supported.")


    def evi_calculation(self,rgb,nir,nir_pred):
        """
        Calculate the EVI loss between two images.

        Parameters:
        rgb (torch.Tensor): The RGB image. Assumes the channel order is RGB.
        nir (torch.Tensor): The NIR image.
        nir_pred (torch.Tensor): The predicted NIR image.

        Returns:
        float: The EVI loss between the NIR and predicted NIR images.
        
        Formula:
        https://en.wikipedia.org/wiki/Enhanced_vegetation_index
        EVI = G * ( (NIR-R) / ( (NIR+C1) * (R-C2) * (Blue + L) )   )    
        Purpose:
        A modification of MSAVI that improves sensitivity in areas with sparse vegetation without
        requiring a soil brightness correction factor.  
        """
        l,c1,c2,g = 1,6,7.5,2.5 # set coefficients
        rgb,nir,nir_pred= self.prepare_tensor_for_loss(rgb,nir,nir_pred)
        
        # extract color bands
        red = rgb[:,0:1,:,:]
        blue = rgb[:,2:3,:,:]
        
        if self.mode=="loss":
            # calculate loss with epsilon to avoid /0
            denom = (nir + c1) * (red - c2) * (blue + l) + 1e-6
            denom_pred = (nir_pred + c1) * (red - c2) * (blue + l) + 1e-6
            evi = g * ((nir - red) / denom)
            evi_pred = g * ((nir_pred - red) / denom_pred)
            # caluclate and return loss
            loss = self.criterion(evi, evi_pred)
            return(loss)
        elif self.mode=="index":
            # calculate without for accurate number/image
            evi = g * ( (nir - red) / ( (nir + c1) * (red - c2) * (blue + l) ) )
            evi_pred = g * ( (nir_pred - red) / ( (nir_pred + c1) * (red - c2) * (blue + l) ) )
            # return evi and evi_pred image/values
            return (evi,evi_pred)
        else:
            raise NotImplementedError(f"Mode '{self.mode}' not implemented. 'loss' or 'index' are supported.")


if __name__ == "__main__":
    # get rs_losses
    nir,nir_pred = torch.rand(1,1,512,512), torch.rand(1,1,512,512)+0.01
    rgb = torch.rand(1,3,512,512)
    
    rs_losses = RemoteSensingIndices() # get object
    # test weighted loss function
    total_loss = rs_losses.get_and_weight_losses(rgb,nir,nir_pred)
    print("Weighted Loss:", total_loss)
