import os
import dnaio
import json, gzip, re
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from collections import defaultdict
from Bio.Seq import Seq
from scipy.optimize import minimize
from .helper import logger


SPATIAL_BARCODE_LEN = 32
BIN_SIZE_TRUE = 30
BIN_SIZE_UNIQUE = 100

CHIP_INFO = {"A": [(8234,63362),
                   (150,20056)
                  ],
             "B": [(78808,133858),
                   (150,20056)
                  ],
             "C": [(149404,204454),
                   (150,20056)
                  ],
             }

def get_chip_info(chip):
    return CHIP_INFO[chip[-2]]

def _rc(x):
    return str(Seq(x).reverse_complement())

def find_neighbors(X_Y_bin):
    x,y = list(map(int,X_Y_bin.split("_")))
    x_range = range(x - 2, x + 3)
    y_range = range(y - 2, y + 3)
    
    return ["_".join([str(i), str(j)]) for i in x_range for j in y_range]

def spatial_umi_count(fq1, fq2, spatial_dir, samplename, **kwargs):

    logger.info("retrieve cell barcode, spatial barcode, spatial umis from reads")
    fh = dnaio.open(fq1, fq2, fileformat="fastq", mode="r")

    spatial_umis = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r1, r2 in fh:
        if len(r2.sequence) >= SPATIAL_BARCODE_LEN:
            cell_bc, umi = r1.name.split('_')[0:2]
            spatial_bc = r2.sequence[0:SPATIAL_BARCODE_LEN]
            spatial_bc_rev = _rc(spatial_bc)
            spatial_umis[cell_bc][umi][spatial_bc_rev] += 1

    fh.close()

    spatial_umis_path = os.path.join(spatial_dir, f'{samplename}_spatial_umis.csv.gz')
    spatial_bc_wl_path = os.path.join(spatial_dir, ".temps", f'{samplename}_spatial_barcode_wl.csv.gz')

    logger.info("prepare whitelist of spatial barcode for HDMI library")
    with gzip.open(spatial_umis_path, 'wt') as f_umis, gzip.open(spatial_bc_wl_path, 'wt') as f_bc:
        f_umis.write(",".join(["Cell_Barcode", "UMI", "Spatial_Barcode", "Read_Count"])+"\n")
        unique_sbs = set()
        for cb, sp_umis in spatial_umis.items():
            line = ''.join([",".join([cb, umi, sb, str(count)]) + "\n"
                            for umi, sb_count in sp_umis.items()
                            for sb, count in sb_count.items()
                            if (unique_sbs.add(sb) or True)])
            f_umis.write(line)
        sb_line = "\n".join(unique_sbs)
        f_bc.write(sb_line)

    # ### metrics
    # with open(os.path.join(spatial_dir, samplename+"_spatial_summary.json"), "w") as fh:
    #     json.dump({"Number_of_Spatial_Reads": total_read},
    #                 fh, 
    #                 indent=4)

def HDMI_process(fq, spatial_dir, samplename, **kwargs):
    logger.info("retrieve valid spatial barcode and coordinates from reads")
    fh = dnaio.open(fq, fileformat="fastq", mode="r")

    HDMI_data = defaultdict(list)
    count = 0
    for r1 in fh:
        search = re.search('R(\d+)C(\d+):(\d+):(\d+) ', r1.name)
        if search:
            tile = search.group(1) + search.group(2)
            x_coor, y_coor = search.group(3), search.group(4)
        else:
            infs = re.split(r'[-_]',r1.name)
            tile_search = re.search('R(\d+)C(\d+)', infs[-2])
            tile = tile_search.group(1) + tile_search.group(2)
            x_coor, y_coor = infs[-1][:4], infs[-1][4:]

        spatial_bc = r1.name[0:SPATIAL_BARCODE_LEN]
        pos = [spatial_bc, tile, x_coor, y_coor]
        HDMI_data[count] = pos

        count +=1
    fh.close()

    df_HDMI = pd.DataFrame.from_dict(HDMI_data, orient='index',
                                    columns=['Spatial_Barcode', 'Tile', 'X_coord', 'Y_coord'])
    barcode_counts = df_HDMI['Spatial_Barcode'].value_counts()
    df_HDMI['Spatial_BC_Count'] = df_HDMI['Spatial_Barcode'].map(barcode_counts)
    df_HDMI_dedup = df_HDMI.drop_duplicates(subset='Spatial_Barcode')

    HDMI_path = os.path.join(spatial_dir, f'{samplename}_spatial_barcode_counts.csv.gz')
    df_HDMI_dedup.to_csv(HDMI_path, index=False, compression="gzip", chunksize=len(df_HDMI)//5)
    
def infer_true_coord(df):
    
    # df['Tile'] = df['Tile'].apply(lambda tile: str(tile))
    df['Tile_x'] = df['Tile'].astype(str).apply(lambda tile: int(tile[1:]))
    df['Tile_y'] = df['Tile'].astype(str).apply(lambda tile: int(tile[0]))

    df['X'] = df['Y_coord'].apply(lambda x: 2176 - int(x))
    df['Y'] = df['X_coord'].astype(int)

    df['X'] = df['X'] + (df['Tile_x'] - 1) * 2178
    df['Y'] = df['Y'] + (df['Tile_y'] - 1) * 4114
    
    df.drop(columns=["Tile", "Tile_x", "Tile_y", "X_coord", "Y_coord"], inplace=True)
    return df

def infer_true_umis(df):

    df["X_bin"] = np.digitize(df['X'], np.arange(0, df['X'].max(), BIN_SIZE_TRUE))
    df["Y_bin"] = np.digitize(df['Y'], np.arange(0, df['Y'].max(), BIN_SIZE_TRUE))
    df['X_Y_bin'] = df['X_bin'].astype(str) + '_' + df['Y_bin'].astype(str)

    df_umis_per_bin = df.groupby(['X_Y_bin']).size().reset_index(name='Count')
    max_umi = 20*df_umis_per_bin['Count'].mean()
    df_umis_per_bin["isAccurate"] = (df_umis_per_bin['Count'] <= max_umi).astype(int)

    df_bins_above_max = df[df['X_Y_bin'].isin(df_umis_per_bin[df_umis_per_bin['Count'] > max_umi]['X_Y_bin'])]
    df_cell_excluded = df_bins_above_max.groupby('X_Y_bin')['Cell_Barcode'].apply(lambda x: x.value_counts().idxmax()).reset_index(name='Cell_Barcode')
    df_spatial_excluded = df[df['Cell_Barcode'].isin(df_cell_excluded['Cell_Barcode'])]

    df = df[(~df['Cell_Barcode'].isin(df_cell_excluded['Cell_Barcode'])) & (~df['Spatial_Barcode'].isin(df_spatial_excluded['Spatial_Barcode']))]

    return (df, df_umis_per_bin)

def find_unique_centers(df):
    
    df["X_bin"] = np.digitize(df['X'], np.arange(0, df['X'].max(), BIN_SIZE_UNIQUE))
    df["Y_bin"] = np.digitize(df['Y'], np.arange(0, df['Y'].max(), BIN_SIZE_UNIQUE))
    df['X_Y_bin'] = df['X_bin'].astype(str) + '_' + df['Y_bin'].astype(str)
    
    
    df_cell_bins = df[['Cell_Barcode', "X_Y_bin"]].groupby(['Cell_Barcode', 'X_Y_bin']).value_counts().reset_index(name='Count')
    cell_bins_dict = df[['Cell_Barcode', "X_Y_bin"]].groupby('Cell_Barcode')['X_Y_bin'].value_counts().to_dict()
    
    #for each bin, count umis in 24 surrounding bins(including itself)
    counts_25_bins = []
    for index, row in df_cell_bins.iterrows():

        bc = row["Cell_Barcode"]
        outter_centers = find_neighbors(row["X_Y_bin"])

        count_outter = 0
        for center in outter_centers:
            center_key = (bc, center)
            if center_key in cell_bins_dict.keys():
                count_outter += cell_bins_dict[center_key]

        counts_25_bins.append(count_outter)

    df_cell_bins["Count_25_bins"] = counts_25_bins
    df_cell_bins.sort_values(['Cell_Barcode','Count','Count_25_bins'], ascending=False, inplace=True)
    
    #for each cell barcode, find bins with top umi count
    df_top_bin = df_cell_bins.groupby('Cell_Barcode').first().reset_index().rename(columns={"Count_25_bins":"Top_bin"}).set_index("Cell_Barcode")
    df_top_bin["Second_bin"] = 0
    
    #surrounding bins info of top bin
    df_bin_25 = df_top_bin.groupby('Cell_Barcode')['X_Y_bin'].apply(lambda x: find_neighbors(x.iloc[0])).explode().reset_index()
    
    #bins that are not top bin 
    df_rest = df_cell_bins.groupby('Cell_Barcode').tail(-1)
    rest_dict = df_rest.groupby('Cell_Barcode')[['X_Y_bin','Count_25_bins']].apply(lambda x: x.values.tolist()).to_dict()
    
    #find second top bin that are not in the surrounding bins of top bin
    for cell, info in rest_dict.items():
        top_bin = df_top_bin.loc[cell, "X_Y_bin"]
        neighbor_bins = find_neighbors(top_bin)
        for bin_count in info:
            bin_coord, count = bin_count
            if bin_coord not in neighbor_bins:
                df_top_bin.loc[cell, "Second_bin"] = count
                break
                
    df_unique_center_bin = df_top_bin[(df_top_bin['Top_bin'] > 1) & (df_top_bin["Top_bin"] >= df_top_bin["Second_bin"] * 2)]
    df_unique_center = df_bin_25[df_bin_25["Cell_Barcode"].isin(df_unique_center_bin.index)]
    df = pd.merge(df, df_unique_center, on=["Cell_Barcode", "X_Y_bin"])
    
    return df

def infer_center_of_cell(df):

    barcode_center = {}

    df_grouped = df.groupby("Cell_Barcode")
    for barcode, df_coords in df_grouped:
        coordinates = df_coords[["X", "Y"]].values
        
        def loss_function(center, data):
            return np.sum(np.linalg.norm(data - center, axis=1))
        
        def constraint_function(x, coordinates):
            x = x.astype(int)
            if np.isin(x, coordinates).all():
                return 0
            else:
                return 100

        initial_center = np.mean(coordinates, axis=0)
        result = minimize(loss_function, initial_center, args=(coordinates,), method='SLSQP', 
                        constraints={'type': 'ineq', 'fun': constraint_function, 'args': (coordinates,)})
        
        optimized_center = list(result.x.astype(int))
        barcode_center[barcode] = optimized_center
    
    df_center = pd.DataFrame.from_dict(barcode_center, orient='index', columns=['X', 'Y']).reset_index(names="Cell_Barcode")
    df_center['RNA_snn_res.0.8'] = 0
    return df_center

def spatial_umi_filtering(rna_dir, spatial_dir, samplename, chip_id, **kwargs):

    logger.info("start filtering spatial umis")
    df_umis = pd.read_csv(f"{spatial_dir}/{samplename}_spatial_umis.csv.gz",
                    compression="gzip")
    df_spatial_coord = pd.read_csv(f"{spatial_dir}/{samplename}_spatial_barcode_counts.csv.gz",
                        compression="gzip")
    df_cells = pd.read_csv(f"{rna_dir}/step3/filtered_feature_bc_matrix/barcodes.tsv.gz", 
                        header=None, names=["Cell_Barcode"])

    summary = {}
    ### calculate saturation
    spatial_barcode_saturation = 1- (len(df_umis)/df_umis["Read_Count"].sum())
    summary["Valid_Spatial_Reads"] = df_umis["Read_Count"].sum()

    df_umis.drop(columns="Read_Count", inplace=True)
    summary["Spatial_Barcode_Saturation"] = spatial_barcode_saturation
    summary["Total_Spatial_UMIs"] = len(df_umis)
    #summary["Ratio_of_Valid_Spatial_Reads"] = f'{(summary["Valid_Spatial_Reads"]/summary["Total_Spatial_Reads"]):.2%}'

    logger.info("get valid spatial umis")
    ### valid spatial barcode
    df_valid_umis = pd.merge(df_umis, df_spatial_coord, on="Spatial_Barcode", how="inner")
    summary["Valid_Spatial_UMIs"] = len(df_valid_umis)
    #summary["Ratio_of_Valid_Spatial_UMIs"] = f'{(summary["Valid_Spatial_UMIs"]/summary["Total_Spatial_UMIs"]):.2%}'
    del df_spatial_coord

    ### unique spatial barcode
    logger.info("get valid spatial umis with unique location")
    df_valid_umis['isUniqueLocation'] = (df_valid_umis['Spatial_BC_Count'] == 1).astype(int)
    df_valid_umis.drop(columns="Spatial_BC_Count", inplace=True)
    summary["Spatial_UMIs_with_Unique_Locations"] = ((df_valid_umis["isUniqueLocation"] == 1)).sum() / len(df_valid_umis)

    ### true spatial umi
    logger.info("infer coordinate of spatial barcode")
    df_valid_umis = infer_true_coord(df_valid_umis)
    x_range, y_range = get_chip_info(chip_id)

    df_valid_umis['X'] = df_valid_umis['X'] - x_range[0]
    df_valid_umis['Y'] = df_valid_umis['Y'] - y_range[0]

    df_true_umis_pre = df_valid_umis[(df_valid_umis['X'] >= 0) & 
                                (df_valid_umis['X'] < (x_range[1]-x_range[0])) & 
                                (df_valid_umis['Y'] >= 0) & 
                                (df_valid_umis['Y'] < (y_range[1]-y_range[0])) & 
                                (df_valid_umis['isUniqueLocation'] == 1)]

    logger.info("find accurate spatial umis")
    df_true_umis, df_umis_per_bin = infer_true_umis(df_true_umis_pre)
    df_valid_umis["isAccurate"] = (df_valid_umis['Cell_Barcode'].isin(df_true_umis['Cell_Barcode']) & 
                                df_valid_umis['Spatial_Barcode'].isin(df_true_umis['Spatial_Barcode'])).astype(int)

    summary["Accurate_Spatial_UMIs"] = (df_valid_umis["isAccurate"].sum() / len(df_valid_umis))
    summary["Accurate_Spatial_UMI_Bins"] = ((df_umis_per_bin["isAccurate"]==1).sum() / len(df_umis_per_bin))

    ### cell barcode from cell calling 
    logger.info("get spatial umis with cells")
    df_valid_umis["isCell"] = (df_valid_umis['Cell_Barcode'].isin(df_cells['Cell_Barcode']) & 
                              (df_valid_umis['isAccurate'] == 1)).astype(int)

    df_umis_cell = (df_valid_umis[df_valid_umis["isCell"] == 1]
                                    .value_counts('Cell_Barcode')
                                    .reset_index())
    summary["Cell-Identified_Spatial_UMIs"] = df_umis_cell["count"].sum() / (df_valid_umis["isAccurate"]==1).sum()
    summary["Mean_Spatial_UMIs_per_Cell"] = int(np.mean(df_umis_cell["count"]))

    ### find unique-center cells      
    logger.info("identify unique-center cells")
    df_unique_center_pre = df_valid_umis[(df_valid_umis["isCell"] == 1) & (df_valid_umis["isAccurate"] == 1)]
    df_unique_center = find_unique_centers(df_unique_center_pre)

    # df_valid_umis["isUniqueCenter"] = (df_valid_umis['Cell_Barcode'].isin(df_unique_center['Cell_Barcode']) & 
    #                     df_valid_umis['Spatial_Barcode'].isin(df_unique_center['Spatial_Barcode'])).astype(int)

    cells_with_unique_centers = len(df_unique_center['Cell_Barcode'].unique())

    assert cells_with_unique_centers != 0
    logger.warning(f"Number of cells with unique-center: {cells_with_unique_centers}")

    ### calculate centers
    logger.info("calculate center of cells")
    df_center = infer_center_of_cell(df_unique_center)
    summary["Number_of_Cells_with_Unique_Center"] = len(df_center)
    summary["Chip_ID"] = chip_id

    ### save csv
    logger.info("save dataframe into CSV format")
    df_valid_umis = df_valid_umis[["Cell_Barcode", "UMI", "Spatial_Barcode", "isUniqueLocation", "isAccurate", "isCell", "X", "Y"]]
    df_valid_umis.to_csv(f"{spatial_dir}/{samplename}_valid_spatial_umis.csv.gz", 
                        index=False, compression="gzip", chunksize=len(df_umis)//5)
    df_unique_center = df_unique_center[["Cell_Barcode", "UMI", "Spatial_Barcode", "X", "Y"]]
    df_unique_center.to_csv(f"{spatial_dir}/{samplename}_spatial_umis_cleaned.csv.gz", 
                        index=False, compression="gzip")
    
    df_center.to_csv(f"{spatial_dir}/{samplename}_cell_locations.csv", index=False)


    ## update summary
    with open(f"{spatial_dir}/.temps/spatialLib/{samplename}_summary.json", "r") as fh:
        summary_step1 = json.load(fh)
    total_spatial_read = summary_step1["stat"]["total"]
    summary = {"Total_Spatial_Reads": total_spatial_read, **summary}

    # summary_final.update(summary)
    summary = {
        k: f'{v:,.2%}' if isinstance(v, float) else v
        for k, v in summary.items()
    }

    df_summary = pd.DataFrame(summary, index=[0])
    df_summary.to_csv(f"{spatial_dir}/{samplename}_spatial_summary.csv", index=False)


