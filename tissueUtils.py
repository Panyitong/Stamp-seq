import json
import cv2
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile, math
import skimage
import skimage.filters.rank as sfr
from .spatialUtils import get_chip_info
from .helper import logger

class tissue_seg:
    def __init__(
        self, DAPI=None, HE=None, Sx="1", Sy="1", Tx="0", Ty="0", degree="0", bins=9, **kwargs
    ):
        self.img = None
        self.HE_TIMG = None
        self.HE = None
        self.blur = None
        self.mask = None

        ## process 记录
        self.threshold = None
        self.gradient = None
        self.edge = None
        self.sum = None
        self.gc = None

        ## defaults
        self.hls_min = 10000
        self.hls_max = 72350
        self.obs_min = 6250

        self.width = 12000
        self.height = 4338
        
        self.width_cut = 4000
        self.height_cut = 1447
        
        if DAPI and HE:
            self.read_tiff(DAPI, "DAPI")
            ##update
            self.transformation(Sx, Sy, Tx, Ty, degree, "DAPI")
            self.cut()
            
            self.read_tiff(HE, "HE")
            self.transformation(Sx, Sy, Tx, Ty, degree, "HE")
            

        elif DAPI:
            self.read_tiff(DAPI, "DAPI")
            ##update
            self.transformation(Sx, Sy, Tx, Ty, degree, "DAPI")
            self.cut()            
        else:
            logger.info("no image provided")
            mask = np.ones((self.height_cut, self.width_cut))
            mask.fill(1)
            self.mask = mask
        
            self.img = np.ones((self.height_cut, self.width_cut, 1))
        
        
        ##DAPI
        self.img = cv2.resize(self.img[:,:], (self.width_cut, self.height_cut), interpolation=cv2.INTER_AREA)
        self.img[self.mask == 0] = 0
        ##HE
        if self.HE is not None:
            ##Thumbil 
            self.HE_TIMG = np.stack([cv2.resize(self.HE[:, :, i], (self.width_cut, self.height_cut), interpolation=cv2.INTER_AREA) for i in range(3)], axis=2)
             
    def read_tiff(self, tiff, stain):
        logger.info(f"read tif image: {stain}")
        img = tifffile.imread(tiff)
        h,w = img.shape[0],img.shape[1]
        if h > w:
            self.height = math.ceil(w * (self.width/h))
        if h < w:
            self.height = math.ceil(h * (self.width/w))
        
        if img.ndim == 3:
            if stain == "DAPI":
                resized_channel = [cv2.resize(img[:,:,0], (self.width, self.height), interpolation=cv2.INTER_AREA)]
            else:
                resized_channel = [cv2.resize(img[:, :, i], (self.width, self.height), interpolation=cv2.INTER_AREA) for i in range(3)]
            
        else:
            resized_channel = [cv2.resize(img[:,:], (self.width, self.height), interpolation=cv2.INTER_AREA)]
        
        if stain == "HE":
            img_resized = np.stack(resized_channel, axis=2)
            self.HE = img_resized
            return 
        else:
            img_resized = np.zeros((self.height, self.width, 3))
            img_resized[:, :, 0] = resized_channel[0]
            img_resized[:, :, 1] = resized_channel[0]
            img_resized[:, :, 2] = resized_channel[0]
        
        img_resized = img_resized.astype("uint8")
        img_cvt = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        cv2.normalize(img_cvt, img_cvt, 255, 0, cv2.NORM_MINMAX)
        
        self.img = img_cvt.astype("uint8")
        self.blur = self._blur()
            
    def transformation(self, Sx, Sy, Tx, Ty, degree, stain):  
        
        logger.info(f"image transformation: {stain}")

        params = [Sx, Sy, Tx, Ty, degree] 
        num_params = len(params[0].split(';'))

        idx = 0

        while idx < num_params:
            params_round = [float(param.split(';')[idx]) for param in params]

            Sx, Sy, Tx, Ty, degree = params_round

            x0 = self.width/2
            y0 = self.height/2
            M_rotate = np.array([[np.cos(np.radians(degree)), -np.sin(np.radians(degree)), x0 * (1 - np.cos(np.radians(degree))) + y0 * np.sin(np.radians(degree))],
                                 [np.sin(np.radians(degree)), np.cos(np.radians(degree)), y0 * (1 - np.cos(np.radians(degree))) - x0 * np.sin(np.radians(degree))],
                                 [0, 0, 1]])

            M_trans = np.array([[Sx, 0, Tx],
                                [0, Sy, Ty],
                                [0, 0, 1]])

            M_f = np.dot(M_trans, M_rotate)

            if stain == "DAPI":
                self.img = cv2.warpPerspective(self.img, M_f, (self.width, self.height))
                self.blur = cv2.warpPerspective(self.blur, M_f, (self.width, self.height))
            elif stain == "HE":
                self.HE = cv2.warpPerspective(self.HE, M_f, (self.width, self.height), borderValue=(255, 255, 255))
        
            idx += 1


    def _blur(self, bins=9):
        return cv2.GaussianBlur(self.img, (bins,bins), 0)

    def _distance(self, mtx, ratio, rev=False):
        return scipy.ndimage.distance_transform_edt(mtx) <= self.height_cut * ratio if rev else scipy.ndimage.distance_transform_edt(mtx) >= self.height_cut * ratio
    
    def _denoise(self, x, size_max, rev=False):

        x = cv2.bitwise_not(x) if rev else x

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(x, 4, cv2.CV_32S)

        areas = stats[1:,cv2.CC_STAT_AREA]
        result = np.zeros((labels.shape), np.uint8)
        for i in range(0, num_labels - 1):
            if areas[i] >= size_max:   #keep
                result[labels == i + 1] = 255

        return cv2.bitwise_not(result) if rev else result
        
    def _threshold(self):
        _, filtered_img = cv2.threshold(self.blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #remove white spot
        filtered_img = self._denoise(filtered_img, self.obs_min)
        #remove black spot
        filtered_img = self._denoise(filtered_img, self.hls_min, rev=True)
        self.threshold = filtered_img.astype("uint8")
        return filtered_img.astype("uint8")
    
    def _gradient(self):
        grad = sfr.gradient(self.blur, skimage.morphology.disk(5))
        grad_norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        grad_rev = cv2.bitwise_not(grad_norm)
        _, filtered_img = cv2.threshold(grad_rev, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        #remove white spot
        filtered_img = self._denoise(filtered_img, self.obs_min)
        #remove black spot
        filtered_img = self._denoise(filtered_img, self.hls_min, rev=True)
        
        self.gradient = filtered_img.astype("uint8")
        return filtered_img.astype("uint8")
    
    def _edge(self):
        h = 0.5 * (np.max(self.blur) - np.min(self.blur)) + np.min(self.blur)
        edge = cv2.Canny(self.blur, 0.1*np.max(self.blur), 0.6*np.max(self.blur))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edge_closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
        _, filtered_img = cv2.threshold(edge_closed, 0, 1, cv2.THRESH_BINARY)
        
        # edge = skimage.feature.canny(self.blur)
        # filtered_img = skimage.morphology.closing(edge)
        filtered_img = filtered_img.astype("uint8")

        filtered_img = self._distance(1 - filtered_img, 0.01, rev=True)
        filtered_img = filtered_img.astype("uint8")

        filtered_img = self._denoise(filtered_img, self.obs_min)
        filtered_img = self._denoise(filtered_img, self.hls_min, rev=True)
        
        self.edge = filtered_img.astype("uint8")
        return filtered_img.astype("uint8")
    
    def cut(self):
        logger.info("tissue segmentation start")
        # prepare smaller version blur
        self.height_cut = math.ceil(self.width_cut * (self.height/self.width))
        self.blur = cv2.resize(self.blur[:,:], (self.width_cut, self.height_cut), interpolation=cv2.INTER_AREA)
        
        filter_comb = np.sum([self._threshold(), self._gradient(), self._edge()], axis=0)
        self.sum = filter_comb

        markers = np.zeros(self.blur.shape, np.uint8)
        trace = np.zeros(self.blur.shape, np.uint8)
        
        ## sure_bg
        sure_bg = np.zeros(self.blur.shape, np.uint8)
        sure_bg[filter_comb == 0] = 1
        sure_bg = self._distance(sure_bg, 0.01)
        
        markers[sure_bg == 1] = 0
        trace[sure_bg == 1] += 1
        
        ## unknown
        unk_bg = np.zeros(self.blur.shape, np.uint8)
        unk_bg[filter_comb >= 1] = 1
        unk_bg = self._denoise(unk_bg, self.hls_max, rev=True)
        
        markers[unk_bg >= 1] = 2
        trace[unk_bg >= 1] += 1
        markers[filter_comb >= 1] = 3
        trace[filter_comb >= 1] += 1

        ## sure_fg
        sure_fg = np.zeros(self.blur.shape, np.uint8)
        sure_fg[filter_comb == 2] = 1
        sure_fg = self._distance(sure_fg, 0.05)
        markers[sure_fg == 1] = 1
        trace[sure_fg == 1] += 1
        
        ## refine fg
        prob_fg = np.zeros(self.blur.shape, np.uint8)
        prob_fg[filter_comb == 3] = 1
        prob_fg = self._distance(sure_fg, 0.025)
        markers[prob_fg == 1] = 1
        trace[prob_fg == 1] += 1

        prob_bg = np.zeros(self.blur.shape, np.uint8)
        prob_bg[filter_comb == 0] = 1
        prob_fg_hls = np.sum([prob_bg, unk_bg], axis=0)

        mtx = np.zeros(self.blur.shape, np.uint8)
        mtx[prob_fg_hls == 2] = 0
        mtx = self._distance(mtx, 0.015)
        mtx = cv2.bitwise_not(mtx.astype("uint8")).astype("uint8")
        markers[mtx == 1] = 2
        markers[trace == 0] = 2
        
        #call
        img = cv2.cvtColor(self.blur.astype("uint8"), cv2.COLOR_GRAY2RGB)
        
        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)

        number_of_iterations = 6
        (mask,_,_) = cv2.grabCut(
            img=img,
            mask=markers,
            rect=None,
            bgdModel=background_model,
            fgdModel=foreground_model,
            iterCount=number_of_iterations,
            mode=cv2.GC_INIT_WITH_MASK,
        )
        self.gc = markers
        self.mask = np.where((mask == 1) | (mask == 3), 1, 0).astype("uint8")

        logger.info("tissue segmentation done")

    def img_processed(self, img_path):
        fig, axes = plt.subplots(nrows=7, ncols=1, figsize=[15, 22])
        axes[0].imshow(self.blur, cmap=plt.cm.gray,interpolation="nearest")
        axes[0].set_title("guassian_blur")        
        axes[1].imshow(self.threshold, cmap=plt.cm.gray, interpolation="nearest")
        axes[1].set_title("treshed")
        axes[2].imshow(self.gradient, cmap=plt.cm.gray, interpolation="nearest")
        axes[2].set_title("gradient")
        axes[3].imshow(self.edge, cmap=plt.cm.gray, interpolation="nearest")
        axes[3].set_title("edge")
        axes[4].imshow(self.sum, cmap=plt.cm.tab10, interpolation="nearest")
        axes[4].set_title("sum")
        axes[5].imshow(self.gc, cmap=plt.cm.tab10, interpolation="nearest")
        axes[5].set_title("gc")
        axes[6].imshow(self.mask, cmap=plt.cm.tab10, interpolation="nearest")
        axes[6].set_title("mask")

        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        return 
    
def mask_bc_under_tissue(mask, center_file, chip_id, bc_csv):
    logger.info("retreive cells under the tissue")
    x_range, y_range = get_chip_info(chip_id)
    x_hdmi = x_range[1] - x_range[0]
    y_hdmi = y_range[1] - y_range[0]

    x_img = mask.shape[1]
    y_img = mask.shape[0]

    scale_mask_x = x_hdmi/x_img
    scale_mask_y = y_hdmi/y_img

    def process_row(row, bc=True):
        new_x = int(row["X"] / scale_mask_x)
        new_y = y_img - int(row["Y"] / scale_mask_y)
        cate = mask[new_y - 1][new_x - 1]

        return row["Cell_Barcode"] if cate == 1 and bc else 1 if cate == 1 else None if bc else 0

    df = pd.read_csv(center_file, usecols=lambda column: column != 'Unnamed: 0')
    bc_under_the_tissue = df.apply(process_row, axis=1).dropna().tolist()

    assert len(bc_under_the_tissue) != 0
    logger.warning(f"Number of cells under tissue coverage: {len(bc_under_the_tissue)}")

    # save bc_under_tissue
    df_bc_under_tissue = pd.DataFrame({'bc_under_the_tissue': bc_under_the_tissue})
    df_bc_under_tissue.to_csv(bc_csv, index=False, header=False)


def parse_alignment_file(**kwargs):
    with open(kwargs["alignment_file"], "r") as fh:
        alignment_params = json.load(fh)

    keys = ["Sx", "Sy", "Tx", "Ty", "degree"]
    msg = "\n"
    for key in keys:
        kwargs[key] = alignment_params[key]
        msg += f"{key}={alignment_params[key]}\n"
    logger.info(msg[:-1])
    return kwargs

def tissue_detection(tissue_dir, spatial_dir, samplename, chip_id, **kwargs) :
    logger.info("start processing tissue")
    ### tissue segmentation
    if kwargs["alignment_file"] is not None:
        tissue_kwargs = parse_alignment_file(**kwargs)
        kwargs.update(tissue_kwargs)
    else:
        keys = ["Sx", "Sy", "Tx", "Ty", "degree"]
        msg = "".join(["\n" + f"{key}={kwargs[key]}" for key in keys])
        logger.info(msg)

    tissue_obj = tissue_seg(**kwargs)

    img_aligned_path_DAPI = f"{tissue_dir}/{samplename}_aligned_DAPI.png"
    cv2.imwrite(img_aligned_path_DAPI, tissue_obj.img)
    if kwargs["DAPI"] is not None:
        img_processed_path = f"{tissue_dir}/{samplename}_processed_DAPI.png"
        tissue_obj.img_processed(img_processed_path)

    if kwargs["HE"] is not None:
        img_aligned_path_HE_TIMG = f"{tissue_dir}/{samplename}_aligned_HE_TIMG.png"
        img_aligned_path_HE = f"{tissue_dir}/{samplename}_aligned_HE.png"

        cv2.imwrite(img_aligned_path_HE_TIMG, cv2.cvtColor(tissue_obj.HE_TIMG, cv2.COLOR_RGB2BGR))
        cv2.imwrite(img_aligned_path_HE, cv2.cvtColor(tissue_obj.HE, cv2.COLOR_RGB2BGR))

    ### mask bc
    bc_csv = f"{tissue_dir}/{samplename}_bc_under_tissue.csv"
    mask_bc_under_tissue(tissue_obj.mask,  
                         f"{spatial_dir}/{samplename}_cell_locations.csv", 
                         chip_id,
                         bc_csv)



  