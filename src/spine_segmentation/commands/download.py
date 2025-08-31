import click, os, pathlib, re
from huggingface_hub import HfApi, hf_hub_download
from .root import cli

REPO_ID = "alexanderdann/CTSpine1K"
VROOT = "raw_data/volumes"
LROOT = "raw_data/labels"

ALIASES = {
    "hnscc": "HNSCC-3DCT-RT",
    "hnscc-3dct-rt": "HNSCC-3DCT-RT",
    "covid": "COVID-19",
    "covid-19": "COVID-19",
    "msd": "MSD-T10",
    "liver": "MSD-T10",
    "msd-t10": "MSD-T10",
    "colonog": "COLONOG",
}

def list_remote_subsets(api: HfApi):
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    subsets = set()
    prefix = f"{VROOT}/"
    for f in files:
        if f.startswith(prefix):
            parts = f.split("/")
            if len(parts) >= 3:
                subsets.add(parts[2])
    return sorted(subsets)

def canonical_subset(name: str, subsets):
    key = name.lower()
    if key in ALIASES:
        return ALIASES[key]
    # try case-insensitive exact match
    for s in subsets:
        if s.lower() == key:
            return s
    # try startswith match (e.g., "hnscc" -> "HNSCC-3DCT-RT")
    for s in subsets:
        if s.lower().startswith(key):
            return s
    return None

@cli.group()
def download():
    """Commands to download dataset(s)."""

@download.command("ctspine1k")
@click.option("--output-dir", default="data/CTSpine1K", show_default=True)
@click.option("--subset", help="One of: HNSCC-3DCT-RT, COVID-19, COLONOG, MSD-T10 (aliases accepted).")
@click.option("--limit", type=int, default=0, show_default=True, help="Download only N cases from the chosen subset.")
@click.option("--list-subsets", is_flag=True, help="List available subsets and exit.")
def download_ctspine1k(output_dir, subset, limit, list_subsets):
    api = HfApi()
    os.makedirs(output_dir, exist_ok=True)

    subsets = list_remote_subsets(api)
    if list_subsets:
        click.echo("Available subsets: " + ", ".join(subsets))
        return

    if subset:
        chosen = canonical_subset(subset, subsets)
        if not chosen:
            raise click.ClickException(f"Subset '{subset}' not found. Try one of: {', '.join(subsets)}")
        # collect volume files for this subset
        vol_prefix = f"{VROOT}/{chosen}/"
        files = [f for f in api.list_repo_files(REPO_ID, repo_type="dataset") if f.startswith(vol_prefix) and f.endswith(".nii.gz")]
        files.sort()
        if limit > 0:
            files = files[:limit]
        if not files:
            raise click.ClickException(f"No volumes found for subset '{chosen}'.")
        click.echo(f"Downloading {len(files)} volumes from subset '{chosen}' ...")
        for vol in files:
            # download volume
            hf_hub_download(REPO_ID, vol, repo_type="dataset", local_dir=output_dir, local_dir_use_symlinks=False)
            # infer label path
            base = pathlib.Path(vol).name[:-7]  # strip ".nii.gz"
            label_name = base + "_seg.nii.gz"  # works for HNSCC, COLONOG, MSD-T10, COVID-19 (_ct_seg)
            label = f"{LROOT}/{chosen}/{label_name}"
            hf_hub_download(REPO_ID, label, repo_type="dataset", local_dir=output_dir, local_dir_use_symlinks=False)
        click.echo("✅ Done.")
    else:
        # full snapshot (huge)
        from huggingface_hub import snapshot_download
        click.echo("No --subset given: downloading the FULL dataset (very large).")
        snapshot_download(REPO_ID, repo_type="dataset", local_dir=output_dir, local_dir_use_symlinks=False)
        click.echo("✅ Done.")
