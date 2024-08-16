from datetime import datetime
import pandas as pd
from math import isnan


ATTRIBUTES = [
    'Similarity',
    'Title',
    'DueDates',
    'Posted',
    'ModifiedDate',
    'CloseDate',
    'Sponsor',
    'SponsorType',
    'Feed',
    'FeedID',
    'ProgramID',
    'AwardType',
    'Eligibility',
    'ApplicantLocation',
    'ApplicantType',
    'CitizenshipReq',
    'ActivityLocation',
    'Status',
    'Amount',
    'MaxAmount',
    'MinAmount',
    'MaxNumAwards',
    'SubmissionDetails',
    'LimitedSubmissionInfo',
    'SubmissionRequirements',
    'CostSharing',
    'RollingDecision',
    'Categories',
    'CFDA',
    'Contacts',
    'URL',
    'SolicitationURL',
    'Description',
    'Prompt',
    'QueryName',
    'Authors'
]

class Raw_Data_Index():
    '''
    Object to handle data wrangling. Find, fetch, extract, parse
    are unique for each data source, and they need to be merged.
    '''
    def __init__(self, filename: str, desc_att: str):
        self.filename = filename
        self.description_attribute = desc_att

    def load_data(self, filename: str):
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
    def date2MMDDYYYY(self, date: str):
        '''
            Each raw data source may have its own data format
        '''
    def mk_empty_row(self):
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
        vals = self.to_dict(row, similarity)
        df = pd.DataFrame({key: vals[key] for key in ATTRIBUTES}, index=[0])
        df['DueDates'] = str(vals['DueDates'])
        return df


class NSF(Raw_Data_Index):
    def __init__(self, filename: str, desc_att: str):
        super().__init__(filename, desc_att)
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.filename)

    def get_descriptions(self):
        return pd.DataFrame({'source': self.__class__.__name__,
                             'filename': self.filename,
                             'row': self.df.index,
                             'description': self.df[self.description_attribute]
                             })

    def date2MMDDYYYY(self, date: str):
        if isinstance(date, float):
            return ''
        if ' ' in date:
            date = date.split(',')[1].strip()
        return datetime.strptime(date, '%Y-%m-%d').strftime('%m/%d/%Y')

    def to_dict(self, idx: int, similarity: float):
        row = self.df.iloc[idx]
        result = self.mk_empty_row()
        result['Similarity'] = similarity
        result['Feed'] = 'NSF'
        result['Title'] = row.Title
        result['Posted'] = self.date2MMDDYYYY(row.Posted_date)
        result['Description'] = row.Synopsis
        result['AwardType'] = row.Award_Type
        result['DueDates'] = [row['Next_due_date']]# double check
        result['CloseDate'] = self.date2MMDDYYYY(row['Next_due_date'])
        result['RollingDecision'] = row['Proposals_accepted_anytime']
        result['ProgramID'] = row.Program_ID
        result['FeedID'] = row.NSF_PD_Num
        result['Status'] = row.Status
        result['URL'] = row.URL
        result['AwardType'] = row.Type
        result['SolicitationURL'] = row.Solicitation_URL
        result['SponsorType'] = 'Federal'
        result['Sponsor'] = 'NSF'
        return result


class SCS(Raw_Data_Index):
    def __init__(self, filename: str, desc_att: str):
        super().__init__(filename, desc_att)
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.filename, lineterminator='\n')  # ^M in data

    def get_descriptions(self):
        return pd.DataFrame({'source': self.__class__.__name__,
                             'filename': self.filename,
                             'row': self.df.index,
                             'description': self.df[self.description_attribute]
                             })

    def date2MMDDYYYY(self, date: str):
        if isinstance(date, float):
            return None
        formats = ['%m/%d/%y', '%Y-%m-%d']
        dt = None
        for f in formats:
            try:
                dt = datetime.strptime(date.strip(), f).strftime('%m/%d/%Y')
                break
            except:
                pass
        if not dt:
            print('scs not a DT', date)
        return dt

    def to_dict(self, idx: int, similarity: float):
        row = self.df.iloc[idx]
        result = self.mk_empty_row()
        result['Similarity'] = similarity
        result['Feed'] = 'SCS Resources Spreadsheet'
        result['Title'] = row['Title']
        result['Sponsor'] = row['Agency/Organization']
        result['SponsorType'] = row['Type']
        result['Posted'] = self.date2MMDDYYYY(row['Post Date'])
        result['CloseDate'] = self.date2MMDDYYYY(row['Due Date'])
        result['URL'] = 'https://docs.google.com/spreadsheets/d/19vQMmH0Vsg0tvf4ia3SBqWTQ8lowQCPhyTOt3hQSVHk/edit?usp=sharing'
        result['Amount'] = row['Amount/Duration']
        result['Description'] = row['Brief Description']
        result['Status'] = 'Open'
        return result


class CMU(Raw_Data_Index):
    def __init__(self, filename: str, desc_att: str):
        super().__init__(filename, desc_att)
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.filename)

    def get_descriptions(self):
        return pd.DataFrame({'source': self.__class__.__name__,
                             'filename': self.filename,
                             'row': self.df.index,
                             'description': self.df[self.description_attribute]
                             })

    def date2MMDDYYYY(self, date: str):
        if isinstance(date, float):
            return None
        return datetime.strptime(date, '%m/%d/%Y').strftime('%m/%d/%Y')

    def to_dict(self, idx: int, similarity: float):
        row = self.df.iloc[idx]
        result = self.mk_empty_row()
        result['Similarity'] = similarity
        result['Feed'] = 'CMU Foundation Relations'
        result['Title'] = row['Opportunity Name']
        result['Sponsor'] = 'NA'  # row['Organization']
        result['SubmissionDetails'] = row['How do I submit a proposal?']
        result['Posted'] = 'NA'
        result['ProgramID'] = row['Solicitation Number']
        result['SponsorType'] = 'NA'  # row['Federal/Non-Federal']
        result['DueDates'] = {'InternalLOI': self.date2MMDDYYYY(row['Internal Letter of Intent Deadline']),
                              'InternalPPD': self.date2MMDDYYYY(row['Internal Pre-Proposal Deadline']),
                              #'NextDueDate':self.date2MMDDYYYY(row['1st Sponsor Deadline']),
                              'FinalDueDate': self.date2MMDDYYYY(row['Final Sponsor Deadline'])
                             }
        result['CloseDate'] = ''
        result['LimitedSubmissionInfo'] = row['CMU Limit']
        result['SubmissionRequirements'] = row['Proposal Requirements (internal, external nominations)']
        result['URL'] = 'https://www.cmu.edu/osp/limited-submissions/index.html'
        result['SolicitationURL'] = row['Website']
        result['Amount'] = 'NA'  # $row['Anticipated Funding Amount']
        result['Description'] = row['Description']
        result['Status'] = 'Open'
        return result


class EXTERNAL(Raw_Data_Index):
    def __init__(self, filename: str, desc_att: str):
        super().__init__(filename, desc_att)
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.filename)

    def get_descriptions(self):
        return pd.DataFrame({'source': self.__class__.__name__,
                             'filename': self.filename,
                             'row': self.df.index,
                             'description': self.df[self.description_attribute]
                            })

    def date2MMDDYYYY(self, date: str):
        if isinstance(date, float):
            return None
        return datetime.strptime(date, '%m/%d/%Y').strftime('%m/%d/%Y')

    def to_dict(self, idx: int, similarity:float):
        # Opportunity Name,Organization,Deadline,Early Career,Description,URL,$ Amount of Award,Duration of Award
        row = self.df.iloc[idx]
        result = self.mk_empty_row()
        result['Similarity'] = similarity
        result['Feed'] = 'CMU External Funding'
        result['Title'] = row['Opportunity Name']
        result['Sponsor'] = row['Organization']
        # result['SubmissionDetails']
        result['Posted'] = 'NA'
        # result['ProgramID'] = row['Solicitation Number']
        result['SponsorType'] = 'External Foundation'#row['Federal/Non-Federal']
        result['DueDates'] = {'Deadline':self.date2MMDDYYYY(row['Deadline'])}
        result['CloseDate'] = self.date2MMDDYYYY(row['Deadline'])
        # result['LimitedSubmissionInfo'] = row['CMU Limit']
        # result['SubmissionRequirements'] = row['Proposal Requirements (internal, external nominations)']
        result['URL'] = 'https://www.cmu.edu/engage/partner/foundations/faculty-staff/index.html'
        result['SolicitationURL'] = row['URL']
        result['Amount'] = row['$ Amount of Award']
        result['Description'] = row['Description']
        result['Status'] = 'Open'
        result['Duration'] = row['Duration of Award']
        result['EarlyCareer'] = row['Early Career']
        return result


class GFORWARD(Raw_Data_Index):
    def __init__(self, filename: str, desc_att: str):
        super().__init__(filename, desc_att)
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.filename)

    def get_descriptions(self):
        return pd.DataFrame({'source': self.__class__.__name__,
                             'filename': self.filename,
                             'row': self.df.index,
                             'description': self.df[self.description_attribute]
                            })

    def date2MMDDYYYY(self, date: str):
        if ':' in date:
            date = date.split(':')[1].strip()
        formats = ['%Y-%m-%d', '%B %d, %Y']
        dt = None
        for f in formats:
            try:
                dt = datetime.strptime(date.strip(), f).strftime('%m/%d/%Y')
                break
            except:
                continue
        if not dt:
            print('gforward not a DT', date)
        return dt

    def to_dict(self, idx: int, similarity: float):
        row = self.df.iloc[idx]
        result = self.mk_empty_row()
        result['Similarity'] = similarity
        result['Feed'] = 'GrantForward'
        result['Title'] = row['Title']
        result['Status'] = row['Status']
        result['Description'] = row['Description']
        result['SolicitationURL'] = row['Source URL']
        result['Sponsor'] = row['Sponsors']
        if not isinstance(row['Deadlines'],float):
            result['DueDates'] = {f'Deadline_{i}':e for i,e in enumerate(row['Deadlines'].split('\n'))}
        else:
            result['DueDates'] = {'Closed':''}
        for k,v in result['DueDates'].items():
            # if isnan(v): continue
            if 'Submission:' in v:
                result['CloseDate'] = self.date2MMDDYYYY(v.split('Submission:')[1])
            if 'Submit Date:' in v:
                result['Posted'] = self.date2MMDDYYYY(v.split('Submit Date:')[1])
        if 'Posted' not in result:
            result['Posted'] = ''
        if 'CloseDate' not in result:
            result['CloseDate'] = ''
        # result['Amount'] = 'Unknown'#row['Amount Info']
        result['MaxAmount'] = row['Maximum Amount']
        result['MinAmount'] = row['Minimum Amount']
        result['AwardType'] = row['Grant Types'].strip()
        result['Eligibility'] = row['Eligibility']
        result['ApplicantLocation'] = row['Applicant Locations']
        result['ActivityLocation'] = row['Activity Locations']
        result['SubmissionDetails'] = row['Submission Info']
        result['ApplicantType'] = row['Applicant Types']
        result['Categories'] = row['Categories']
        result['Contacts'] = row['Contacts']
        result['DueDates']['Submit Date'] = self.date2MMDDYYYY(row['Submit Date'])
        result['ModifiedDate'] = self.date2MMDDYYYY(row['Modified Date'])
        result['URL'] = row['GrantForward URL']
        result['CitizenshipReq'] = row['Citizenships']
        result['MaxNumAwards'] = row['Maximum Number of Awards']
        result['MinNumAwards'] = row['Minimum Number of Awards']
        result['LimitedSubmissionInfo'] = row['Limited Submission Info']
        result['CostSharing'] = row['Cost Sharing']
        result['CFDA'] = row['CFDA Numbers']
        result['FeedID'] = row['GrantForward URL'].split('?grant_id=')[1]#https://www.grantforward.com/grant?grant_id=186134
        return result


class GRANTS(Raw_Data_Index):
    def __init__(self, filename: str, desc_att: str):
        super().__init__(filename, desc_att)
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.filename)

    def get_descriptions(self):
        return pd.DataFrame({'source': self.__class__.__name__,
                             'filename': self.filename,
                             'row': self.df.index,
                             'description': self.df[self.description_attribute]
                            })

    def date2MMDDYYYY(self, date: str):
        if isinstance(date, datetime):
            return date.strftime('%m/%d/%Y')
        if isinstance(date, float):
            if isnan(date):
                return ''
            else:
                date = str(int(date))
        dt = None
        for f in ['%m%d%Y', '%Y-%m-%d']:
            try:
                dt = datetime.strptime(date, f).strftime('%m/%d/%Y')
            except:
                pass
        if not dt:
            print('stumped!', date)
        return dt

    def to_dict(self, idx: int, similarity: float):
        # https://apply07.grants.gov/help/html/help/index.htm#t=XMLExtract%2FXMLExtract.htm
        row = self.df.iloc[idx]
        result = self.mk_empty_row()
        result['Similarity'] = similarity
        result['Feed'] = 'Grants.gov'
        result['FeedID'] = row['OpportunityID']
        result['Title'] = row['OpportunityTitle']
        result['ProgramID'] = row['OpportunityNumber']
        gg_oppcat = {'D': 'Discretionary',
                     'M': 'Mandatory',
                     'C': 'Continuation',
                     'E': 'Earmark',
                     'O': 'Other'
                    }
        result['Categories'] = gg_oppcat[row['OpportunityCategory']]
        gg_fundinsttype = {'G': 'Grant',
                           'CA': 'Cooperative Agreement',
                           'O': 'Other',
                           'PC': 'Procurement Contract'
                           }
        result['AwardType'] = gg_fundinsttype[row['FundingInstrumentType']]
        # CategoryOfFundingActivity
        # CategoryExplanation
        result['CFDA'] = row['CFDANumbers']
        if row['EligibleApplicants'] in GG_ELIGAPP:
            result['ApplicantType'] = GG_ELIGAPP[row['EligibleApplicants']]
        result['Eligibility'] = row['AdditionalInformationOnEligibility']
        # AgencyCode
        result['Sponsor'] = row['AgencyName']
        if isnan(row['PostDate']):
            result['Posted'] = ''
        else:
            result['Posted'] = self.date2MMDDYYYY(str(int(row['PostDate'])))
        result['DueDates'] = {}
        result['CloseDate'] = self.date2MMDDYYYY(row['CloseDate'])
        if isnan(row['LastUpdatedDate']):
            result['ModifiedDate'] = ''
        else:
            result['ModifiedDate'] = self.date2MMDDYYYY(row['CloseDate'])
        result['MaxAmount'] = row['AwardCeiling']
        result['MinAmount'] = row['AwardFloor']
        result['Amount'] = row['EstimatedTotalProgramFunding']
        result['MaxNumAwards'] = row['ExpectedNumberOfAwards']
        result['Description'] = row['Description']
        # Version
        result['CostSharing'] = row['CostSharingOrMatchingRequirement']
        # GrantorContactEmailDescription
        result['Contacts'] = {'Email': row['GrantorContactEmail'],
                              'Contact': row['GrantorContactText'],
                              'Name': row['GrantorContactName'],
                              'Phone': row['GrantorContactPhoneNumber']
                              }
        result['URL'] = f'https://www.grants.gov/search-results-detail/{result["FeedID"]}'  # row['AdditionalInformationURL']
        result['SolicitationURL'] = row['AdditionalInformationURL']
        # AdditionalInformationText
        # CloseDateExplanation
        # OpportuintyCategoryExplanation
        # FiscalYear
        # EstimatedSynopsisCloseDateExplanation
        return result
 

class PIVOT(Raw_Data_Index):
    def __init__(self, filename: str, desc_att: str):
        super().__init__(filename, desc_att)
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.filename)
    def get_descriptions(self):
        return pd.DataFrame({'source': self.__class__.__name__,
                             'filename': self.filename,
                             'row': self.df.index,
                             'description': self.df[self.description_attribute]
                             })

    def date2MMDDYYYY(self, date: str):
        if 'sponsor' in date:
            return ''
        return datetime.strptime(date.strip(), '%d %b %Y').strftime('%m/%d/%Y')

    def to_dict(self, idx: int, similarity: float):
        row = self.df.iloc[idx]
        result = self.mk_empty_row()
        result['Similarity'] = similarity
        result['Feed'] = 'Proquest PIVOT'
        result['FeedID'] = row['Ex Libris Pivot-RP ID']
        if len(row['Title'].split('Funder: '))>1:
            result['Sponsor'] = row['Title'].split('Funder: ')[1]
            result['Title'] = row['Title'].split('Funder: ')[0]
        else:
            result['Title'] = row['Title']
            result['Sponsor'] = row['Funder']
        result['ProgramID'] = row['Funder ID']
        # Funder location
        result['SponsorType'] = row['Funder type']
        if not isinstance(row['Upcoming deadlines'],float):
            result['DueDates'] = {f'Deadline_{i}':e for i,e in enumerate(row['Upcoming deadlines'].split('\n'))}
        else:
            result['DueDates'] = {}
            # Note
        for k,v in result['DueDates'].items():
            if 'sponsor' in str(v):
                result['CloseDate'] = self.date2MMDDYYYY(v.split(' - ')[0])
                if 'Posted' not in result:
                    result['Posted'] = self.date2MMDDYYYY(v.split(' - ')[0])
        result['Eligibility'] = row['Eligibility']
        result['ApplicantLocation'] = row['Applicant/Institution Location']
        result['CitizenshipReq'] = row['Citizenship']
        result['ActivityLocation'] = row['Activity location']
        result['ApplicantType'] = row['Applicant type']
        # Career stage
        result['Description'] = row['Abstract']
        result['URL'] = 'https:'+row['Link to Pivot-RP'].split(' ')[1]
        result['SolicitationURL'] = 'https:'+row['Website'].split(' ')[1]
        result['Categories'] = row['Keywords']
        result['AwardType'] = row['Funding type']
        result['MaxAmount'] = row['Amount Upper']
        result['Amount'] = row['Amount']
        # Amount Note
        # Related Funders
        # Alternate Title
        result['CFDA'] = row['CFDA Numbers']
        # Related Programmes
        # Notice
        return result


class SAM(Raw_Data_Index):
    def __init__(self, filename: str, desc_att: str):
        super().__init__(filename, desc_att)
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.filename)

    def get_descriptions(self):
        return pd.DataFrame({'source': self.__class__.__name__,
                             'filename': self.filename,
                             'row': self.df.index,
                             'description': self.df[self.description_attribute]
                             })

    def date2MMDDYYYY(self, date: str):
        if isinstance(date, float):
            if isnan(date):
                return ''
            else: print('stumped', date)
        if '.' in date:
            date = date.split('.')[0].strip()
        formats = ['%Y-%m-%d %H:%M:%S-%U',
                   '%Y-%m-%d',
                   '%Y-%m-%dT%H:%M:%S-%U:%W',
                   '%Y-%m-%d %H:%M:%S',
                   '%Y-%m-%dT%H:%M:%S',
                   '%Y-%m-%dT%H:%M+%S:%U',
                   '%Y-%m-%dT%H:%M%:%S+%U:%W',
                   '%Y-%m-%dT%H:%M:%S+%U:%W'
                   ]
        dt = None
        for f in formats:
            try:
                dt = datetime.strptime(date, f).strftime('%m/%d/%Y')
                break
            except:
                pass
        if not dt:
            print('stumped!', date)
        return dt

    def to_dict(self, idx: int, similarity: float):
        row = self.df.iloc[idx]
        result = self.mk_empty_row()
        result['Similarity'] = similarity
        result['Feed'] = 'SAM.gov'
        result['FeedID'] = row['NoticeId']
        result['Title'] = row['Title']
        result['ProgramID'] = row['Sol#']
        result['Sponsor'] = row['Department/Ind.Agency']
        # CGAC
        # Sub-Tier
        # FPDS Code
        # Office
        # AAC Code
        result['Posted'] = self.date2MMDDYYYY(row['PostedDate'])
        result['AwardType'] = row['Type']
        # BaseType
        # ArchiveType
        result['DueDates'] = {'ArchiveDate': self.date2MMDDYYYY(row['ArchiveDate']),
                              'ResponseDeadLine': self.date2MMDDYYYY(row['ResponseDeadLine']),
                              'AwardDate': self.date2MMDDYYYY(row['AwardDate'])
                              }
        result['CloseDate'] = self.date2MMDDYYYY(row['ResponseDeadLine'])
        # SetASideCode
        # SetASide
        # NaicsCode
        # ClassificationCode
        # PopStreetAddress
        # PopCity
        # PopState
        result['ActivityLocation'] = row['PopZip']
        # Pop Country
        result['Status'] = row['Active']
        # AwardNumber
        result['Amount'] = row['Award$']
        # Awardee
        result['Contacts'] = {'Title': row['PrimaryContactTitle'],
                              'Name': row['PrimaryContactFullname'],
                              'Email': row['PrimaryContactEmail'],
                              'Phone': row['PrimaryContactPhone'],
                              'Fax': row['PrimaryContactFax']
                              }
        # SecondaryContactTitle, SecondaryContactFullname, SecondaryContactEmail, SecondaryContactPhone,SecondaryConteactFax
        result['SponsorType'] = row['OrganizationType']
        # State, City, ZipCode, CountryCode
        result['SolicitationURL'] = row['AdditionalInfoLink']
        result['URL'] = row['Link']
        result['Description'] = row['Description']
        return result


class ARXIV(Raw_Data_Index):
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

    def date2MMDDYYYY(self, date: str):
        if isinstance(date, float):
            if isnan(date):
                return ''
            else: print('stumped', date)
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
            except:
                pass
        if not dt:
            print('stumped!', date)
        return dt

    def to_dict(self, idx: int, similarity: float):
        row = self.df.iloc[idx]
        result = self.mk_empty_row()
        result['Similarity'] = similarity
        result['Feed'] = 'arxiv.org'
        result['FeedID'] = row['id']
        result['Title'] = row['title']
        result['ProgramID'] = row['categories']
        result['Sponsor'] = 'NA'  # row['Department/Ind.Agency']
        # CGAC
        # Sub-Tier
        # FPDS Code
        # Office
        # AAC Code
        result['Posted'] = self.date2MMDDYYYY(row['version_created'])
        result['AwardType'] = 'NA'  # row['Type']
        # BaseType
        # ArchiveType
        #result['DueDates'] = {'ArchiveDate': self.date2MMDDYYYY(row['ArchiveDate']),
                              #'ResponseDeadLine': self.date2MMDDYYYY(row['ResponseDeadLine']),
                              #'AwardDate': self.date2MMDDYYYY(row['AwardDate'])
                              #}
        result['CloseDate'] = self.date2MMDDYYYY(row['last_update'])
        # SetASideCode
        # SetASide
        # NaicsCode
        # ClassificationCode
        # PopStreetAddress
        # PopCity
        # PopState
        #result['ActivityLocation'] = row['PopZip']
        # Pop Country
        result['Status'] = row['journal_ref']
        # AwardNumber
        #result['Amount'] = row['Award$']
        # Awardee
        #result['Contacts'] = {'Title': row['PrimaryContactTitle'],
                              #'Name': row['PrimaryContactFullname'],
                              #'Email': row['PrimaryContactEmail'],
                              #'Phone': row['PrimaryContactPhone'],
                              #'Fax': row['PrimaryContactFax']
                              #}
        # SecondaryContactTitle, SecondaryContactFullname, SecondaryContactEmail, SecondaryContactPhone,SecondaryConteactFax
        #result['SponsorType'] = row['OrganizationType']
        # State, City, ZipCode, CountryCode
        result['SolicitationURL'] = row['doi']
        result['URL'] = f'https://arxiv.org/abs/{row["id"]}'
        result['Description'] = row['abstract']
        result['Authors'] = row['authors']
        return result
