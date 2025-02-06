'''
Function to help parse idiosyncrasies of raw data sources

Create a new class for each raw data source (e.g. arXiv)
'''
from math import isnan
from datetime import datetime
import pandas as pd

ATTRIBUTES = [
    'Similarity',
    'Title',
    'Posted',
    'CloseDate',
    'Sponsor',
    'Feed',
    'FeedID',
    'ProgramID',
    'AwardType',
    'Status',
    'URL',
    'SolicitationURL',
    'Description',
    'Authors'
]


class RawDataIndex():
    '''
    Object to handle data wrangling. Find, fetch, extract, parse
    are unique for each data source, and they need to be merged.
    '''
    def __init__(self, filename: str, desc_att: str):
        self.filename = filename
        self.description_attribute = desc_att

    def load_data(self):
        '''
            Read in data from files, which have their own unique requirements
        '''
    def get_descriptions(self):
        '''
            Return which attribute should be used for embedding and subsequent
            similarity to user prompts
        '''
    def print(self, row: int, similarity: float):
        '''
            Visualize this element if it is returned
        '''
    def print_title(self, row: int, similarity: float):
        '''
            Print only the title to stdout if saving output to csv
        '''
    def date_to_mmddyyyy(self, date: str):
        '''
            Each raw data source may have its own data format
        '''
    def mk_empty_row(self):
        '''
            Create an empty row to be filled with data from the raw file
        '''
        return {k: None for k in ATTRIBUTES}

    def to_csv(self, row: int, similarity: float):
        """ Convert the data to a pandas DataFrame

        Args:
            row (int): The index of the row in the raw file to convert
            similarity (float): The similarity score of the description
                                to the prompt

        Returns:
            df (pd.DataFrame): DataFrame from key:value pairs
        """


class ARXIV(RawDataIndex):
    '''
    Class to handle arxiv data
    '''
    def __init__(self, filename: str, desc_att: str):
        super().__init__(filename, desc_att)
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.filename, quotechar='"')

    def get_descriptions(self):
        return pd.DataFrame({'source': self.__class__.__name__,
                             'filename': self.filename,
                             'row': self.df.index,
                             'description': self.df[self.description_attribute]
                             })

    def date_to_mmddyyyy(self, date: str):
        if isinstance(date, float):
            if isnan(date):
                return ''
            else:
                print('stumped', date)
        if '.' in date:
            date = date.split('.')[0].strip()
        formats = ['%a, %d %b %Y %H:%M:%S %Z',
                   '%Y-%m-%d'
                   ]
        dt = None
        for f in formats:
            try:
                dt = datetime.strptime(date, f).strftime('%m/%d/%Y')
                break
            except ValueError:
                pass
        if not dt:
            print('stumped!', date)
        return dt

    def to_dict(self, idx: int, similarity: float):
        '''
            Convert the raw data to a dictionary entry with relevant info
        '''
        row = self.df.iloc[idx]
        result = self.mk_empty_row()
        result['Similarity'] = similarity
        result['Feed'] = 'arxiv.org'
        result['FeedID'] = row['id']
        result['Title'] = row['title']
        result['ProgramID'] = row['categories']
        result['Sponsor'] = 'NA'
        result['Posted'] = self.date_to_mmddyyyy(row['version_created'])
        result['AwardType'] = 'NA'
        result['CloseDate'] = self.date_to_mmddyyyy(row['last_update'])
        result['Status'] = row['journal_ref']
        result['SolicitationURL'] = row['doi']
        if len(str(row['id'])) < 10:  # hack to handle strange 0 issue in url
            result['URL'] = f'https://arxiv.org/abs/{row["id"]}0'
        else:
            result['URL'] = f'https://arxiv.org/abs/{row["id"]}'
        result['Description'] = row['abstract']
        result['Authors'] = row['authors']
        return result
