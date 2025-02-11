import os
import pandas as pd
import lzstring
from ..utils.plotUtil import plotUtils
from ..utils.report.websummaryRNA import websummaryRNA

data = {}

def line_color(p):
    _color = ['#dbdbf4',
             '#c5c5f5',
             '#afaff6',
             '#9999f7',
             '#8383f8',
             '#6d6df9',
             '#5858fb',
             '#4141fc',
             '#2b2bfd',
             '#1616fe']
    if p==1:
        return '#0000ff'
    elif p==0:
        return '#dddddd'
    else:
        return _color[int(p/0.1)]



def diff_table(f, N=20):
    df = (pd.read_table(f)
         .assign(cluster=lambda df: df['cluster'].map(lambda x: f'cluster{x}'))
    )
    # table_id, classes
    fold_change_title = 'avg_logFC'
    if 'avg_log2FC' in df.columns:
        fold_change_title = 'avg_log2FC'
    df2 = (df.drop(columns=['p_val',  'pct.1',  'pct.2'])
        .groupby('cluster')
        .apply(lambda x: x.sort_values(['p_val_adj', fold_change_title], ascending=[True, False]).head(N))
        .reset_index(drop=True)
        .pivot_table(values=[fold_change_title, 'p_val_adj'], index=['Ensembl', 'gene'], columns='cluster')
        .swaplevel(axis=1)
        .sort_index(1, 0, key=lambda x: x.str.replace('cluster', '').astype(int))
    )
    # index_names
    return (df2.to_html(table_id='marker_table', classes='display', na_rep='-', index_names=False)
        .replace('border="1"', ''))

def reduction_data(reduction_umi):
    df = pd.read_table(reduction_umi, index_col=0)
    df = df.drop(['orig.ident', 'nFeature_RNA', 'percent.mito'], axis=1)
    data['reduction'] = df.to_dict(orient='list')
    data['params'] = [ _ for _ in df.columns if _.startswith(('RNA_snn_res','seurat_clusters')) ]
    data['labs'] = df.columns[0:2].to_list()


def report(samplename, outdir, **kwargs):
    import json
    from jinja2 import Environment, FileSystemLoader

    summary_file = os.path.join(outdir, f'{samplename}_summary.json')
    assert os.path.exists(summary_file), f'{summary_file} not found!'
    with open(summary_file) as fh:
        summary = json.load(fh)

    sequencing_table = {}
    sequencing_table['Number of Reads'] = f'{summary["stat"]["total"]:,}'
    sequencing_table['Valid Barcodes'] = f'{summary["stat"]["valid"]/summary["stat"]["total"]:.2%}'
    sequencing_table['Sequencing Saturation'] = f'{summary["cells"]["Sequencing Saturation"]:.2%}'
    del summary["cells"]["Sequencing Saturation"]

    if 'no_anchor' in summary["stat"]:
        sequencing_table['Without Anchor'] = f'{summary["stat"]["no_anchor"]:,}'

    if 'trimmed' in summary["stat"]:
        sequencing_table['Trimmed'] =  f'{summary["stat"]["trimmed"]:,}'
    if 'too_short' in summary["stat"]:
        sequencing_table['Too Short'] =  f'{summary["stat"]["too_short"]:,}'
    b_total_base = sum([sum(v) for v in summary["barcode_q"].values()])
    b30_base = sum([sum(v[30:]) for v in summary["barcode_q"].values()])
    sequencing_table['Q30 Bases in Barcode'] = f'{b30_base/b_total_base:.2%}'
    u_total_base = sum([sum(v) for v in summary["umi_q"].values()])
    u30_base = sum([sum(v[30:]) for v in summary["umi_q"].values()])
    sequencing_table['Q30 Bases in UMI'] = f'{u30_base/u_total_base:.2%}'

    mapping_table = {k: f'{v:.2%}' for k, v in summary["mapping"].items()}

    cells_table = dict([(k, f'{v:,}') if isinstance(v, int) else (k,f'{v:.2%}') for k,v in summary["cells"].items()])
    
    sample_table = {
        'Name': samplename, 
        'Description': '',
        'Transcriptome': summary["reference"],
        'Chemistry': summary["stat"].get("chemistry", "custom"),
        'Include introns': summary["include_introns"],
        'Seeksoul tools Version': summary["__version__"]
    }
    
    reduction_xls = os.path.join(outdir, 'step4', 'tsne_umi.xls')
    assert os.path.exists(reduction_xls), f'{reduction_xls} not found!'
    reduction_data(reduction_xls)

    cells_gz = os.path.join(outdir, 'step3/filtered_feature_bc_matrix/barcodes.tsv.gz')
    barcodes_gz = os.path.join(outdir, 'step3/raw_feature_bc_matrix/barcodes.tsv.gz')
    mtx_gz = os.path.join(outdir, 'step3/raw_feature_bc_matrix/matrix.mtx.gz')
    barcod_rank_data = plotUtils.barcode_rank_plot_rna(cells_gz, barcodes_gz, mtx_gz)
    data["barcode_rank_data"]=barcod_rank_data

    f =  os.path.join(outdir, 'step4', 'FindAllMarkers.xls')
    marker_table = diff_table(f)

    header=('Samplename,Estimated_Number_of_Cells,Mean_Reads_per_Cell,Median_Genes_per_Cell,Number_of_Reads,'
            'Valid_Barcodes,Sequencing_Saturation,Reads_Mapped_Confidently_to_Genome,Fraction_Reads_in_Cells,'
            'Total_Genes_Detected,Median_UMI_Counts_per_Cell')

    summary_data = [
             samplename,
             cells_table['Estimated Number of Cells'],
             cells_table['Mean Reads per Cell'],
             cells_table['Median Genes per Cell'],
             sequencing_table['Number of Reads'],
             sequencing_table['Valid Barcodes'],
             sequencing_table['Sequencing Saturation'],
             mapping_table['Reads Mapped Confidently to Genome'],
             cells_table['Fraction Reads in Cells'],
             cells_table['Total Genes Detected'],
             cells_table['Median UMI Counts per Cell']
           ]

    with open(os.path.join(outdir, f'{samplename}_summary.csv'), 'w') as fh:
        fh.write(header + '\n')
        fh.write(','.join(str(_).replace(',', '') for _ in summary_data)+ '\n')

    object = websummaryRNA(
        summary_json=f"{outdir}/{samplename}_summary.json",
    )

    data_json = object.to_json(
        diff_table=f"{outdir}/step4/FindAllMarkers.xls",
        dim_table=f"{outdir}/step4/tsne_umi.xls",
        filtered_dir=f"{outdir}/step3/filtered_feature_bc_matrix",
        raw_dir=f"{outdir}/step3/raw_feature_bc_matrix",
    )
    template_dir_new = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/report'))
    env = Environment(loader=FileSystemLoader(template_dir_new))
    template = env.get_template('base.html')
    with open(os.path.join(outdir, f'{samplename}_report.html'), 'w') as fh:
        fh.write(template.render(websummary_json_data=data_json))

