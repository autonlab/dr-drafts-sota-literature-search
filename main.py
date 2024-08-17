"""
Dr. Draft's SOTA Literature Search

Command-line Arguments:
    - `-p, --prompt`: Description of the work you want to do (default: 'CLI')
    - `-k, --k`: Number of matches to return (default: 3)
    - `-o, --output`: CSV file to store output
    - `-t, --title`: Title for results if multiple queries (default: 'CLI prompt')

Usage:
    python main.py [-p PROMPT] [-k K] [-a] [-o OUTPUT] [-t TITLE] [-i] [-s]

Example:
    python main.py -p "Research on climate change" -k 5 -o results.csv -t "Climate Change Research"
"""
import faulthandler
from argparse import ArgumentParser
from warnings import filterwarnings
from src import sota_search

IDIR = './index'
EMBEDDINGS = f'{IDIR}'+'/embeddings.pkl'

if __name__ == "__main__":
    faulthandler.enable()
    filterwarnings('ignore')
    p = ArgumentParser()
    p.add_argument('-p', '--prompt', default='CLI',
                   help='Description of the work you want to do')
    p.add_argument('-k', '--k', default=3, type=int,
                   help='Restrict search to arXiv papers submitted within last s(int) years.')
    p.add_argument('-o', '--output',
                   help='CSV file to store output')
    p.add_argument('-t', '--title', default='CLI prompt',
                   help='Title for results if multiple queries')
    args = p.parse_args()

    sota_search.show_flags(args.k,
                              args.prompt,
                              args.output,
                              args.title
                              )

    experiment = sota_search.Experiment(args.prompt, EMBEDDINGS, args.k)
    experiment.run()
    results = experiment.select_results(range(args.k))
    if len(results)<args.k:
        results = experiment.select_results(range(10*args.k))

    results.drop_duplicates(subset=['Title'],
                            keep='first',
                            inplace=True,
                            ignore_index=True
                            )
    if not args.output:
        sota_search.results2console(results.iloc[:args.k])
    else:
        sota_search.results2csv(results.iloc[:args.k], args.output, args.prompt, args.title)

