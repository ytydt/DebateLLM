import os
import json
import tarfile
import zipfile
import shutil
from pathlib import Path
import subprocess
import requests


def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return path


def download_mmlu(dest: Path = Path('data/datasets/mmlu')) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = _download('https://people.eecs.berkeley.edu/~hendrycks/data.tar', dest / 'data.tar')
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest)
    tar_path.unlink()
    data_dir = dest / 'data'
    if data_dir.exists():
        for item in data_dir.iterdir():
            shutil.move(str(item), dest)
        data_dir.rmdir()


def download_usmle(dest: Path = Path('data/datasets/usmle')) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    base = 'https://huggingface.co/datasets/medalpaca/medical_meadow_usmle_self_assessment/raw/main/'
    files = [
        'question_with_images.json',
        'step1.json', 'step1_solutions.json',
        'step2.json', 'step2_solutions.json',
        'step3.json', 'step3_solutions.json',
    ]
    for fname in files:
        _download(base + fname, dest / fname)


def download_medmcqa(dest: Path = Path('data/datasets/medmcqa')) -> None:
    import gdown
    dest.mkdir(parents=True, exist_ok=True)
    out_path = Path('data.zip')
    gdown.download(id='15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky', output=str(out_path), quiet=False)
    with zipfile.ZipFile(out_path) as z:
        z.extractall(dest)
    out_path.unlink()


def download_medqa(dest: Path = Path('data/datasets/medqa')) -> None:
    import gdown
    dest.mkdir(parents=True, exist_ok=True)
    out_path = Path('data_clean.zip')
    gdown.download(id='1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw', output=str(out_path), quiet=False)
    with zipfile.ZipFile(out_path) as z:
        z.extractall(dest)
    shutil.move(dest / 'data_clean' / 'questions' / 'US', dest / 'questions')
    shutil.rmtree(dest / 'data_clean')
    out_path.unlink()


def download_pubmedqa(dest: Path = Path('data/datasets/pubmedqa')) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    url1 = 'https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json'
    url2 = 'https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json'
    ori = _download(url1, dest / 'ori_pqal.json')
    gt = _download(url2, dest / 'test_ground_truth.json')

    with open(ori) as f:
        data1 = json.load(f)
    with open(gt) as f:
        data2 = json.load(f)
    train, test = {}, {}
    for k, v in data1.items():
        if k in data2:
            test[k] = v
        else:
            train[k] = v
    with open(dest / 'dev.json', 'w') as f:
        json.dump(train, f, indent=4)
    with open(dest / 'test.json', 'w') as f:
        json.dump(test, f, indent=4)


def download_cosmosqa(dest: Path = Path('data/datasets/cosmosqa')) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'kaggle', 'datasets', 'download',
        'thedevastator/cosmos-qa-a-large-scale-commonsense-based-readin',
        '-p', str(dest), '--unzip'
    ], check=True)


def download_gpqa(dest: Path = Path('data/datasets/gpqa')) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = _download('https://github.com/idavidrein/gpqa/blob/main/dataset.zip?raw=true', dest / 'dataset.zip')
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest, pwd=b'deserted-untie-orchid')
    for f in (dest / 'dataset').glob('*.csv'):
        shutil.move(str(f), dest)
    shutil.rmtree(dest / 'dataset')
    zip_path.unlink()


def download_all():
    download_mmlu()
    download_usmle()
    download_medmcqa()
    download_medqa()
    download_pubmedqa()
    download_cosmosqa()
    download_gpqa()


if __name__ == '__main__':
    download_all()
