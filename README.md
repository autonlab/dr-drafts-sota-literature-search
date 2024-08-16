# arxiv_context_search
Provide your best description of an abstract for a paper you are looking for, and this tool computes cosine similarity with embeddings of arxiv papers, returns the top k matches, and provides urls. This has a cli interface.

## Installation
```
git clone git@github.com:autonlab/arxiv_context_search.git
cd arxiv_context_search/
conda create --name searcharxiv --file env.txt
conda activate searcharxiv
./test.sh
```


## How to Run
```
python main.py -k 5 -o results.csv -p 'We propose to research new methods for verification and validation of artificial intelligence and machien learning models fit to data.'
```
-- or create a file with a prompt you can edit with a text editor --
```
./prompts/sample_prompt.sh
```
Command-line Arguments:
    - `-p, --prompt`: Description of the work you want to do (default: 'CLI')
    - `-k, --k`: Number of matches to return (default: -1)
    - `-a, --active`: Restrict search to CFPs that have not expired (default: False)
    - `-o, --output`: CSV file to store output
    - `-t, --title`: Title for results if multiple queries (default: 'CLI prompt')
    - `-s, --summary`: Prints AI-generated summary of paper abstracts (default: False)


#### Embedding all arxiv abstracts takes a while. If you want a progress bar for multi-gpu indexing:
Edit your site-package file for SentenceTransformers.py by adding:
```
from tqdm import tqdm
```

And add a tqdm function call around line 502-503 (depending on where you place import statement):
```
results_list = sorted([output_queue.get() for _ in tqdm(range(last_chunk_id))], key=lambda x: x[0])
```
This will throw warnings if a different conda environment uses the same site-package.
