"""
Module for the state of the art (SOTA) literature search
"""
from os.path import exists
from os import environ
import textwrap
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ansi2html import Ansi2HTMLConverter
from flask import request

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


def is_running_in_flask():
    """Check if the code is running in a Flask context"""
    return request and request.url_rule is not None


def platform_specific_text(text: str):
    """Convert ANSI to HTML if running in Flask, otherwise return ANSI text"""
    if is_running_in_flask():
        return Ansi2HTMLConverter().convert(text)
    else:
        return text


def results2console(results: pd.DataFrame):
    """Print the results of the SOTA literature search to the console

    Args:
        results (pd.DataFrame): The results of the SOTA literature search
        print_summary (bool, optional): Defaults to False.
    """
    show_testometer_banner()
    show_prizes()
    print(platform_specific_text(
        f"""\n*** Dr. Draft\'s ({DRDRAFT}) top {len(results)} picks***""")
    )
    for i in range(len(results)):
        x = results.iloc[i]
        show_prize_banner(f'{x.Title}', x.Similarity)
        show_one('URL', x['URL'])
        abstract_desc = x['Description']
        show_one('Abstract', abstract_desc, limit=True)


def results2csv(results: pd.DataFrame, output_fn: str,
                prompt: str, qname: str):
    """ Write the results of the SOTA Literature Search to a CSV file

    Args:
        results (pd.DataFrame): The results of the SOTA Literature Search
        output_fn (str): The filename for the output CSV
        prompt (str): The prompt that generated these results
        qname (str): The name of the query
    """
    show_testometer_banner()
    show_prizes()
    print(platform_specific_text(
        f'\n*** Dr. Draft\'s ({DRDRAFT}) top {len(results)} picks ***')
    )
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


def show_prize_banner(message: str, prize: float, show_score=False,
                      limit=True):
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
        print(platform_specific_text(
            f'\033[1;38;5;{color}m{header} {text[len(header)-1:]}\033[0m')
        )
    else:
        print(platform_specific_text(
            f'\033[1;38;5;{color}m{text[len(header):]}\033[0m')
        )


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

    print(platform_specific_text(
        f'\033[1m{key1}:\033[0m\033[38;5;8m{text[len(header)-1:]}\033[0m')
    )


def encode_prompt(prompt):
    """Encode a prompt using the {DRDRAFT} model

    Args:
        prompt (str): The prompt to encode

    Returns:
        Array: Vector representation of the prompt
    """
    model = SentenceTransformer(DRDRAFT)
    return model.encode([prompt])


def read_narrative_embeddings(filename: str):
    """ Read narrative embeddings from a file

    Args:
        filename (str): The filename to read

    Returns:
        Pandas.DataFrame: The narrative embeddings
    """
    return pd.read_pickle(filename)


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


def compute_similarity(embedded_prompt: np.array,
                       embedded_narratives: np.array):
    """ Compute the similarity between a prompt and a set of narratives """
    similarity = [_[0] for _ in cosine_similarity(
            embedded_narratives.iloc[:, 4:],
            embedded_prompt.reshape(1, -1)
        )
    ]
    return pd.DataFrame(
        {'similarity': similarity},
        index=embedded_narratives.index
    )


def sort_similarity_descending(similarity: pd.DataFrame):
    """ Sort the similarity in descending order """
    return similarity.sort_values('similarity', inplace=False, ascending=False)


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
    """Show a color-coded tier list for the SOTA Literature Search

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
        print(platform_specific_text(
            f' - \033[38;5;{color}m{metric}\033[0m -- {pname}')
        )


def show_testometer_banner():
    """Show a color banner for the SOTA Literature Search

    Args:
        None

    Returns:
        None: Prints to console
    """
    print(platform_specific_text(''))
    show_prize_banner(
        "Dr. Draft's SOTA Literature Search!",
        0.99
    )
    show_one(
        'Has someone published something similar to your idea before?',
        "Let's find out!"
    )


def show_prompt(prompt: str):
    """Show the prompt supplied for the SOTA Literature Search

    Args:
        prompt (str): The prompt supplied by the user

    Returns:
        None: Prints to console
    """
    print(platform_specific_text(f'\033[38;5;84m\nPrompt:\033[0m {prompt}'))


def show_flags(k: int, prompt: str, output: str, title: str):
    """Show the flags supplied for the SOTA Literature Search

    Args:
        k (int): Number of matches to return
        output (str): CSV file to store output
        title (str): Title for results if multiple queries
    """

    print(platform_specific_text("\033[38;5;84m\nSPECIFICATION: \033[0m"))
    print(platform_specific_text(
        f'Search for {k} most cosine-similar abstracts to \"{title}\" prompt:')
    )
    show_prompt(prompt)
    if output:
        print(platform_specific_text(f' - Results will be saved to {output}'))


def show_data_stats(ds):
    """Show statistics about the data

    Args:
        ds (pd.DataFrame): The data

    Returns:
        None: Prints to console
    """
    print(platform_specific_text(f' - Searching {len(ds)} opportunities:'))
    feeds = []
    for source in ds.filename.unique():
        f = source.split('_')[0].split('/')[-1]
        if f not in feeds:
            feeds.append(f)
    for f in feeds:
        print(platform_specific_text(
            f'   -- {f}: {len(ds[ds.filename.str.contains(f)])} opportunities')
        )
