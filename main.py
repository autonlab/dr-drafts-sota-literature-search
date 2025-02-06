"""
Dr. Draft's SOTA Literature Search

Command-line Arguments:
    - `-p, --prompt`: Description of the work you want to do
                        (default: 'CLI')
    - `-k, --k`: Number of matches to return
                        (default: 3)
    - `-o, --output`: CSV file to store output
    - `-t, --title`: Title for results if multiple queries
                        (default: 'CLI prompt')

Usage:
    python main.py [-p PROMPT] [-k K] [-a] [-o OUTPUT] [-t TITLE] [-i] [-s]

Examples:

    CLI
    $ python main.py -p "Research on climate change"
        -k 5 -o results.csv -t "Climate Change Research"

    Flask
    $ python server.py
"""
import faulthandler
from argparse import ArgumentParser
from warnings import filterwarnings
import io
import sys
from os import environ
import pandas as pd

from src import sota_search
from src import data as DATA

environ["TOKENIZERS_PARALLELISM"] = "false"  # parallel GPU throws warning
N_TIERS = 11
PRIZES_RGB = [240, 245, 250, 255, 46, 33, 92, 226, 202, 199]
PRINTMAXCHARS = 80
PRINTMAXLINES = 12
TARGET = {'NSF': 'Synopsis',
          'SCS': 'Brief Description',
          'SAM': 'Description',
          'GRANTS': 'Description',
          'GFORWARD': 'Description',
          'CMU': 'Summary',
          'PIVOT': 'Abstract',
          'EXTERNAL': 'Description',
          'ARXIV': 'abstract'
          }
DRDRAFT = 'all-mpnet-base-v2'
DRGIST = 'facebook/bart-large-cnn'


IDIR = './index'
EMBEDDINGS = f'{IDIR}'+'/embeddings.pkl'

NARRATIVE_EMBEDDINGS = None


def select_topk_results(df: pd.DataFrame, k: int):
    '''   Select top k results from the dataframe. '''
    df = pd.DataFrame([read_neighbor(df, i) for i in range(k)])
    df['CloseDate'] = pd.to_datetime(df['CloseDate'])
    return df


def read_neighbor(near_neighbors: pd.DataFrame, i: int):
    """ Read data for neighbor i from disk. """
    x = NARRATIVE_EMBEDDINGS.loc[near_neighbors.index[i]]
    record_from_disk = getattr(DATA, x.source)(x.filename, TARGET[x.source])
    return record_from_disk.to_dict(x.row, near_neighbors.iloc[i].similarity)


def load_data(filename: str = EMBEDDINGS):
    '''    Load data from disk. '''
    global NARRATIVE_EMBEDDINGS
    if NARRATIVE_EMBEDDINGS is None:
        NARRATIVE_EMBEDDINGS = sota_search.read_narrative_embeddings(filename)
        sota_search.show_data_stats(NARRATIVE_EMBEDDINGS)


def run_dr_drafts(prompt: str, k: int, output_filename: str, title: str):
    ''' Run Dr. Drafts SOTA Literature Search. '''
    if sota_search.is_running_in_flask():
        output_capture = io.StringIO()
        sys.stdout = output_capture

    sota_search.show_flags(k, prompt, output_filename, title)
    embedded_prompt = sota_search.encode_prompt(prompt)
    if NARRATIVE_EMBEDDINGS is None:
        load_data(EMBEDDINGS)
    similarity = sota_search.compute_similarity(
        embedded_prompt, NARRATIVE_EMBEDDINGS
    )
    nearest_neighbors = sota_search.sort_similarity_descending(similarity)

    results = select_topk_results(nearest_neighbors, k)
    results.drop_duplicates(subset=['Title'],
                            keep='first',
                            inplace=True,
                            ignore_index=True)
    sota_search.results2console(results.iloc[:k])
    if sota_search.is_running_in_flask():
        sys.stdout = sys.__stdout__
        return output_capture.getvalue()


if __name__ == "__main__":
    faulthandler.enable()
    filterwarnings('ignore')
    p = ArgumentParser()
    p.add_argument('-p', '--prompt', default='CLI',
                   help='Description of the abstract you want to see')
    p.add_argument('-k', '--k', default=3, type=int,
                   help='Number of matches to return')
    p.add_argument('-o', '--output',
                   help='CSV file to store output')
    p.add_argument('-t', '--title', default='CLI prompt',
                   help='Title for results if multiple queries')
    args = p.parse_args()

    run_dr_drafts(args.prompt, args.k, args.output, args.title)
