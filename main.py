"""
The script takes command-line arguments to customize the analysis and output. It uses the `proposal_meter` module to perform the analysis and generate the results.

Command-line Arguments:
    - `-p, --prompt`: Description of the work you want to do (default: 'CLI')
    - `-k, --k`: Number of matches to return (default: -1)
    - `-s, --since`: Restrict search to arXiv abstracts submitted within the last s years (default: -1)
    - `-o, --output`: CSV file to store output
    - `-t, --title`: Title for results if multiple queries (default: 'CLI prompt')

Usage:
    python main.py [-p PROMPT] [-k K] [-a] [-o OUTPUT] [-t TITLE] [-i] [-s]

Example:
    python main.py -p "Research on climate change" -k 5 -o results.csv -t "Climate Change Research" -s 3
"""
import faulthandler
from argparse import ArgumentParser
from warnings import filterwarnings
from src import proposal_meter

IDIR = './index'
EMBEDDINGS = f'{IDIR}'+'/embeddings.pkl'

if __name__ == "__main__":
    faulthandler.enable()
    filterwarnings('ignore')
    p = ArgumentParser()
    p.add_argument('-p', '--prompt', default='CLI',
                   help='Description of the work you want to do')
    p.add_argument('-k', '--k', default=-1, type=int,
                   help='Number of matches to return')
    p.add_argument('-s', '--since', default=-1, type=int,
                   help='Restrict search to arXiv papers submitted within last s(int) years.')
    p.add_argument('-o', '--output',
                   help='CSV file to store output')
    p.add_argument('-t', '--title', default='CLI prompt',
                   help='Title for results if multiple queries')
    args = p.parse_args()

    proposal_meter.show_flags(args.k,
                              args.prompt,
                              args.since,
                              args.output,
                              args.title
                              )

    experiment = proposal_meter.Experiment(args.prompt, EMBEDDINGS, args.k)
    experiment.run()
    results = experiment.select_results(range(args.k),args.since)
    if len(results)<args.k:
        results = experiment.select_results(range(10*args.k),args.since)

    results.drop_duplicates(subset=['Title'],
                            keep='first',
                            inplace=True,
                            ignore_index=True
                            )
    if not args.output:
        proposal_meter.results2console(results.iloc[:args.k])
    else:
        proposal_meter.results2csv(results.iloc[:args.k], args.output, args.prompt, args.title)

