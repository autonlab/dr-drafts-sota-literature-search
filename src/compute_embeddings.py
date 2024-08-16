"""
Compute embeddings for narratives using SentenceTransformer.

Args:
    IDIR (str): index director path to directory containing pickled CFPs/FOAs.
    data_files: Files containing CFP/FOA data. Split by get_*.sh scripts
Returns:
    index_directory/embeddings.pkl
"""
from typing import List
from sys import argv
from glob import glob
from sentence_transformers import SentenceTransformer
import pandas
import torch
import data as DATA_CLASSES

MODEL_NAME = 'all-mpnet-base-v2'
DESCRIPTION_ATTR = {
                    'ARXIV': 'abstract'
                    }


def encode_narratives(N: List[str]) -> pandas.DataFrame:
    """Encode narratives using SentenceTransformer. Multi-GPU support.

    Model is set to all-mpnet-base-v2.

    Args:
        N (List[str]): List of narratives to encode. Descriptions of CFPs/FOAs.

    Returns:
        pandas.DataFrame: DataFrame with #narratives x #dims.
    """
    transformer = SentenceTransformer(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        tds = ['cuda:1', 'cuda:2', 'cuda:0', 'cuda:3']
        pool = transformer.start_multi_process_pool(target_devices=tds)
        embs = transformer.encode_multi_process(N,
                                                pool,
                                                batch_size=1024,  # 128
                                                chunk_size=len(N)/1000  # 100
                                                )
        transformer.stop_multi_process_pool(pool)
    else:
        embs = transformer.encode(N,
                                  show_progress_bar=True,
                                  batch_size=64,
                                  device=device
                                  )
    ncols = len(embs[0])
    attnames = [f'F{i}' for i in range(ncols)]
    return pandas.DataFrame(embs, columns=attnames)


def glob2objects(glob_pattern: str):
    """Convert globbed files to objects.

    Args:
        glob_pattern (str): Which files to include

    Returns:
        List[obj]: A list of class objects for reading each raw data files
    """
    files = list(glob(glob_pattern))
    classes = [f.split('/')[-1].split('_')[0] for f in files]
    zset = zip(files, classes)
    print('zset',zset)
    objs = [getattr(DATA_CLASSES, c)(f, DESCRIPTION_ATTR[c]) for f, c in zset]
    print('obj',objs)
    return objs


def objects2descriptions(Objs: list):
    """Convert objects to descriptions.

    Args:
        objects (list): List of class objects

    Returns:
        pandas.DataFrame: DataFrame with descriptions read from objects
    """
    return pandas.concat([obj.get_descriptions() for obj in Objs],
                         ignore_index=True)


if __name__ == "__main__":
    IDIR = argv[1]
    objects = glob2objects(f'{IDIR}/*_S*')
    descriptions = objects2descriptions(objects)
    df = descriptions.drop_duplicates(
        subset=['description'],
        keep='last',
        ignore_index=True
        )
    #df = descriptions
    if not torch.cuda.is_available():
        print('Warning: No GPU detected. Using CPU.')
    embeddings = encode_narratives(df.description.astype(str))
    result = pandas.concat([df, embeddings], axis=1)
    result.to_pickle(f'{IDIR}/embeddings.pkl')
