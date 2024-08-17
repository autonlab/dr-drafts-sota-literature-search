# Dr Draft's state-of-the-art (SOTA) Literature Search
Provide your best description of an abstract for a paper you are looking for, and Dr Draft (a sentence transformer) will compute cosine similarity with embeddings of arxiv paper abstracts, returns the top k matches, and provides urls. This has a cli interface and can output to file for easy sharing of results with others.

When working, this code should automatically download, parse, and embed arxiv papers.  You can add more data to the stack, so long as it matches the naming convention FEED_S000 (e.g. ARXIV_S135).  Each feed gets sliced into smaller files.  Take a look at ARXIV_S000 to get an idea for the format in which you can add data from other sources.  The dream would be to integrate publishers but, naturally, all publishers do not make their data available for bulk download.


## Installation
```
git clone https://github.com/autonlab/dr-drafts-sota-literature-search.git
cd arxiv_context_search/
conda env create -f env.yml
conda activate drdraft
./test.sh
```


## How to Run
```
python main.py -k 5 -o results.csv -p 'We propose to research new methods for verification and validation of artificial intelligence and machien learning models fit to data.'
```
-- or create a file with a prompt you can edit with a text editor --
```
./prompts/cli_sample.sh
./prompts/csv_sample.sh
```
Command-line Arguments:
    - `-p, --prompt`: Description of the work you want to do (default: 'CLI')
    - `-k, --k`: Number of matches to return (default: -1)
    - `-o, --output`: CSV file to store output
    - `-t, --title`: Title for results if multiple queries (default: 'CLI prompt')


#### Embedding all 2.5M+ arxiv abstracts takes over an hour. If you want a progress bar for multi-gpu indexing:
Edit your site-package file for SentenceTransformers.py by adding:
```
from tqdm import tqdm
```

And add a tqdm function call around line 502-503 (depending on where you place import statement):
```
results_list = sorted([output_queue.get() for _ in tqdm(range(last_chunk_id))], key=lambda x: x[0])
```
This will throw warnings if a different conda environment uses the same site-package.
