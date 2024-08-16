"""
Module for the Proposal Test-O-Meter
"""
import textwrap
import pandas as pd
import time
from os.path import exists
from os import environ
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from src import data as DATA
from functools import lru_cache


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
DRGRANT = 'all-mpnet-base-v2'
DRGIST = 'facebook/bart-large-cnn'


def results2console(results: pd.DataFrame, print_summary=False):
    """Print the results of the Proposal Test-O-Meter to the console

    Args:
        results (pd.DataFrame): The results of the Proposal Test-O-Meter
        print_summary (bool, optional): Defaults to False.
    """
    show_testometer_banner()
    show_prizes()
    print(f"""\n*** Dr. Grant\'s ({DRGRANT}) top {len(results)} picks***""")
    for i in range(len(results)):
        x = results.iloc[i]
        show_prize_banner(f'{x.Title}', x.Similarity)
        show_one('URL', x['URL'])
        description = x['Description']
        show_one('Abstract', description, limit=True)
        

def results2csv(results: pd.DataFrame, output_fn: str, prompt: str, qname: str):
    """ Write the results of the Proposal Test-O-Meter to a CSV file

    Args:
        results (pd.DataFrame): The results of the Proposal Test-O-Meter
        output_fn (str): The filename for the output CSV
        prompt (str): The prompt that generated these results
        qname (str): The name of the query
    """
    show_testometer_banner()
    show_prizes()
    print(f'\n*** Dr. Grant\'s ({DRGRANT}) top {len(results)} picks ***')
    for i in range(len(results)):
        x = results.iloc[i]
        show_prize_banner(f'{x.Title}', x.Similarity,
                          show_score=True, limit=False)
    results['Prompt'] = prompt
    results['QueryName'] = qname
    results['Eligibility'] = 'See URL'
    results['ApplicantLocation'] = 'See URL'
    results['ActivityLocation'] = 'See URL'
    results['SubmissionDetails'] = 'See URL'
    results.to_csv(output_fn, index=False, mode='a',
                   header=not exists(output_fn))


def show_prize_banner(message: str, prize: float, show_score=False, limit=True):
    """ Print a color-coded prize banner to the console

    Args:
        message (str): The message to display
        prize (float): The prize value
        show_score (bool, optional): Defaults to False.
        limit (bool, optional): Defaults to True.
    """
    header = f'[{prize:0.4f}] '
    color = PRIZES_RGB[int(prize*100)//N_TIERS]

    clean_val1 = message.replace("'", "\'").replace("\n", " -- ")
    text = header+clean_val1
    if limit:
        text = '\n'.join(
            textwrap.wrap(text, PRINTMAXCHARS, break_long_words=True))
    if show_score:
        print(f'\033[1;38;5;{color}m{header} {text[len(header)-1:]}\033[0m')
    else:
        print(f'\033[1;38;5;{color}m{text[len(header):]}\033[0m')


def show_one(key1: str, val1: str, limit=False):
    """Print a formatted key-value pair to the console

    Args:
        key1 (str): Bolded text for the key
        val1 (str): Grey text for the value
        limit (bool): Whether to limit the number of lines printed
    """
    header = f'{key1}: '
    clean_val1 = val1.replace("'", "\'").replace("\n", " -- ")
    text = header + clean_val1
    if limit:
        text = '\n'.join(textwrap.wrap(text,
                                       PRINTMAXCHARS,
                                       break_long_words=True,
                                       max_lines=PRINTMAXLINES))
    else:
        text = '\n'.join(textwrap.wrap(text,
                                       PRINTMAXCHARS,
                                       break_long_words=True))
    print(f'\033[1m{key1}:\033[0m\033[38;5;8m{text[len(header)-1:]}\033[0m')


def description(ds, nearest_neighbors, i):
    """ Print a description from the dataset

    Args:
        ds (Pandas.DataFrame): The dataset
        nearest_neighbors (List): Sorted list of nearest neighbors
        i (int): The neighbor to print
    """
    fn = ds.loc[nearest_neighbors.index[i]].filename
    row = ds.loc[nearest_neighbors.index[i]].row
    source = fn.split('/')[-1].split('_')[0]
    funcname = eval(source)
    raw_data = funcname(fn, TARGET[source])
    show_one(i, raw_data.df.loc[row].Description)


def encode_prompt(prompt):
    """Encode a prompt using the {DRGRANT} model

    Args:
        prompt (str): The prompt to encode

    Returns:
        Array: Vector representation of the prompt
    """
    model = SentenceTransformer(DRGRANT)
    return model.encode([prompt])

@lru_cache()
def read_narrative_embeddings(filename: str, ttl_hash=None):
    """ Read narrative embeddings from a file

    Args:
        filename (str): The filename to read

    Returns:
        Pandas.DataFrame: The narrative embeddings
    """
    del ttl_hash
    return pd.read_pickle(filename)

def get_ttl_hash(seconds=600):
    """https://stackoverflow.com/questions/31771286/python-in-memory-cache-with-time-to-live"""
    """Return the same value withing `seconds` time period"""
    return round(time.time() / seconds)


def sort_by_similarity_to_prompt(prompt, embedded_narratives):
    """ Sort a set of narratives by similarity to a prompt

    Args:
        prompt (str): The prompt to compare
        embedded_narratives (pandas.DataFrame): The embedded narratives

    Returns:
        Pandas.DataFrame: The sorted narratives
    """
    embedded_prompt = encode_prompt(prompt)
    similarity = [_[0] for _ in
                  cosine_similarity(embedded_narratives.iloc[:, 4:],
                                    embedded_prompt.reshape(1, -1))]
    result = pd.DataFrame({'similarity': similarity},
                          index=embedded_narratives.index)
    result.sort_values('similarity', inplace=True, ascending=False)
    return result


def human_readable_dollars(num: float):
    """Convert a number of dollars to a human-readable string

    Args:
        num (float): Number of dollars

    Returns:
        str: Human-readable string e.g. '1.2M'
    """
    for unit in ('', 'K', 'M', 'B'):
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}'
        num /= 1024.0
    return f'{num:.1f}T'


def show_prizes():
    """Show a color-coded tier list for the Proposal Test-O-Meter

    Args:
        None: Uses hard-coded values for prizes and colors

    Returns:
        None: Prints to console
    """
    prizes = ['poor fish, try again!', 'clammy', 'harmless', 'mild',
              'naughty,  but nice', 'Wild', 'Burning!', 'Passionate!!',
              'Hot Stuff!!!', 'UNCONTROLLABLE!!!!']
    for pidx in reversed(range(len(prizes))):
        color = PRIZES_RGB[pidx]
        low_lim = pidx/len(prizes)
        hi_lim = (pidx+1)/len(prizes)
        pname = prizes[pidx]
        metric = f'Cosine Similarity in [{low_lim:.1f},{hi_lim:.1f})'
        print(f' - \033[38;5;{color}m{metric}\033[0m -- {pname}')


def show_testometer_banner():
    """Show a color banner for the Proposal Test-O-Meter

    Args:
        None

    Returns:
        None: Prints to console
    """
    print()
    show_prize_banner(
        "Dr. Grant's Proposal Test-O-Meter!",
        0.99
    )
    show_one(
        'How attractive is your idea to potential sponsors?',
        "Let's find out!"
    )


def show_prompt(prompt: str):
    """Show the prompt supplied for the Proposal Test-O-Meter

    Args:
        prompt (str): The prompt supplied by the user

    Returns:
        None: Prints to console
    """
    print(f'Prompt: {prompt}')


def show_flags(k: int, prompt: str, since: bool, output: str, title: str):
    """Show the flags supplied for the Proposal Test-O-Meter

    Args:
        k (int): Number of matches to return
        since (int): Restrict search to papers within the last s years
        output (str): CSV file to store output
        title (str): Title for results if multiple queries
    """

    print('\033[38;5;84m\nSPECIFICATION: \033[0m')
    print(f"""Search for {k} most cosine-similar funding opportunity descriptions based on the
          "{title}" prompt:""")
    show_prompt(prompt)
    if since>0:
        print(f' - Restricting search to papers submitted within the last {since} years.')
    if output:
        print(f' - Results will be saved to {output}')


def show_data_stats(ds):
    """Show statistics about the data

    Args:
        ds (pd.DataFrame): The data

    Returns:
        None: Prints to console
    """
    print(f' - Searching {len(ds)} opportunities:')
    feeds = []
    for source in ds.filename.unique():
        feed = source.split('_')[0].split('/')[-1]
        if feed not in feeds:
            feeds.append(feed)
    for feed in feeds:
        print(f'   -- {feed}: {len(ds[ds.filename.str.contains(feed)])} opportunities')
    print(' - \033[38;5;202mData Sources Last Updated: 08/16/2024\033[0m')


class Experiment():
    """ Class for running
    """
    def __init__(self, prompt: str, embeddingsFN: str, k: int):
        self.prompt = prompt
        self.embeddingsFN = embeddingsFN
        self.embeddings = None
        self.nearest_neighbors = None
        self.k = k
        self.since = -1
    def run(self):
        """ Run the experiment
        """
        self.embeddings = read_narrative_embeddings(self.embeddingsFN,ttl_hash=get_ttl_hash())
        show_data_stats(self.embeddings)
        self.nearest_neighbors = sort_by_similarity_to_prompt(self.prompt, self.embeddings)

    def select_results(self, neighbors, since=-1):
        df = pd.DataFrame([self.read_neighbor(i) for i in neighbors])
        df['CloseDate'] = pd.to_datetime(df['CloseDate'])
        if since>0:
            df.dropna(subset=['CloseDate'],inplace=True)
            df = df[~(df['CloseDate'] < (datetime.now()-relativedelta(years=since)))]
        return df

    def read_neighbor(self, i):
        x=self.embeddings.loc[self.nearest_neighbors.index[i]]
        return getattr(DATA,x.source)(x.filename,TARGET[x.source]).to_dict(x.row,self.nearest_neighbors.iloc[i].similarity)
