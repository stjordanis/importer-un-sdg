#!/usr/bin/env python
# coding: utf-8

# # UN Sustainable Development Goals
# 
# This notebook implements the pre-processing needed for importing the UN SDG dataset into OWID's grapher database.
# A rough outline of the process:
# 
#   1. **Load the dataset exported from the UN SDG Indicators database website**
#   1. **Replace the codes used in dimensions with their full names from the codelists**
#   1. **Export the entities (countries) used in the dataset**
#   1. **Export all the data as datasets, variables and datapoints**

# In[1]:


import pandas as pd
import numpy as np
import collections
import itertools
import functools
import requests
import json
import math
from datetime import datetime
from tqdm import tqdm

from import_metadata import extract_description

pd.options.display.max_columns = None
pd.set_option('display.max_colwidth', -1)


# ## Load the data
# 
# The data was obtained from the [UN SDG Indicators database](https://unstats.un.org/sdgs/indicators/database). We selected all _Goals_ (topmost category in the classification of indicators) and requested the entire dataset. 

# In[2]:


def str_to_float(s):
    try:
        # Parse strings with thousands (,) separators
        return float(s.replace(',','')) if type(s) == str else s
    except ValueError:
        return None


# In[3]:


original_df = pd.read_csv(
    "data/20190903150325064_drifter4e@gmail.com_data.csv", 
    converters={'Value': str_to_float},
    low_memory=False
)


# Remove entries for which no value exists.

# In[4]:


original_df = original_df[original_df['Value'].notnull()]


# ## Structure & plan
# 
# Some terminology:
# 
# - **Dimensions**: The column names wrapped in square brackets (`[]`) are _dimensions_ of the dataset. For example, `[Age]` is a dimension.
# - **Dimension entities**: A value in a dimension column is a _dimension entity_. For example, `15-24` is an entity within the `[Age]` dimension.
# 
# In addition, there is a _geographical_ dimension and a _time_ dimension in this dataset – those are special dimensions and we won't call them "dimensions", only columns wrapped in `[]` will be called dimensions.
# 
# How we will organise the exports:
# 
# - For each unique `SeriesCode` we will create a **dataset**.
# - For each `SeriesCode` and a unique combination of dimension entities we will construct a **variable**. So for example, `(SeriesCode: SI_POV_EMP1, Age: 15-14, Sex: MALE)` will form one variable. Remember, `Age` and `Sex` are dimensions, `15-24` and `MALE` are dimension entities. If a some `SeriesCode` doesn't have any corresponding dimensions (they are `NaN`) then there is only a single variable for that `SeriesCode`.
# - For each variable, we will create a **datapoints** file that contains the values broken down by geography and year. So it would have 3 columns: `geo`, `year` and `value`. The variable id will be derived from the filename (`datapoints_[variable_id]`).
# - We will create an **entities** file containing all possible geo entities.
# 

# In[5]:


# A section of the dataset that contains Age and Sex dimension entities
original_df[original_df['[Age]'].notnull() & original_df['[Sex]'].notnull()]


# ## Replace column codes with full names
# 
# Some dimension values in the dataset are codes like `EROSN` or `CHESP` which refer to _erosion_ and _chemical spill_, respectively. **Codelists** contain the mappings from **codes** → **full names**.
# 
# The codelists can be retrieved from the SDGAPI: https://unstats.un.org/SDGAPI/swagger/#!/Goal/V1SdgGoalDataGet
# 
# Specifically, the `GET /v1/sdg/Goal/Data` API call.

# In[6]:


metadata = requests.get('https://unstats.un.org/SDGAPI/v1/sdg/Goal/Data').json()


# There are codelists for dimensions (each column that is wrapped in `[]`) and codelists for the `Nature` and `Units` columns.

# In[7]:


codelists_by_dimension_name = { 
    codelist['id']: { 
        row['code']: row['description'] for row in codelist['codes'] 
    } for codelist in metadata['dimensions'] 
}


# In[8]:


codelists_by_column_name = { 
    codelist['id']: { 
        row['code']: row['description'] for row in codelist['codes'] 
    } for codelist in metadata['attributes'] 
}


# #### Check which dataset dimensions we have codelists for
# 
# There are currently no codelists for `Bounds`. We will have keep this in mind and avoid overwriting that column.

# In[9]:


dimension_names = [c[1:-1] for c in original_df.columns if c[0] == '[' and c[-1] == ']']


# In[10]:


comparison = pd.merge(
    pd.DataFrame({ 'dimensions': dimension_names }), 
    pd.DataFrame({ 'codelists': list(codelists_by_dimension_name.keys()) }), 
    left_on='dimensions', right_on='codelists', how='left')

comparison['match'] = comparison['dimensions'] == comparison['codelists']


# In[11]:


comparison


# #### Replace the codes for each column with the full names from the codelists

# In[12]:


# Returns a dataframe with the dimension codes mapped to full names
def expand_codes(df, codelists_by_dimension_name, codelists_by_column_name):
    
    df = df.copy()
    dimension_names = [c[1:-1] for c in df.columns if c[0] == '[' and c[-1] == ']']
    codelist_names = list(codelists_by_dimension_name.keys())
    dimensions_to_remap = list(set(dimension_names) & set(codelist_names))
    
    # Remap the dimensions (the columns wrapped in [])
    for dim in dimensions_to_remap:
        col = '[' + dim + ']'
        df[col] = df[col].map(codelists_by_dimension_name[dim])
    
    column_names = [c for c in df.columns if c not in set(dimension_names)]
    codelist_names = list(codelists_by_column_name.keys())
    columns_to_remap = list(set(column_names) & set(codelist_names))
    
    # Remap the other columns (Nature & Units, and maybe others)
    for col in columns_to_remap:
        df[col] = df[col].map(codelists_by_column_name[col])
        
    return df


# In[13]:


expanded_df = expand_codes(original_df, codelists_by_dimension_name, codelists_by_column_name)


# #### Check that it was successful

# In[14]:


expanded_df[['[Hazard type]']].drop_duplicates()


# In[15]:


expanded_df[['Units']].drop_duplicates()


# ## Export the geographical entities (countries & regions)
# 
# Store all geographical entities (countries, regions & aggregates) in a separate file. They should be internally consistent within the dataset, but the final Our World in Data standardisation will happen in a later step.

# In[16]:


entities = expanded_df[['GeoAreaCode', 'GeoAreaName']]     .sort_values(by='GeoAreaCode')     .drop_duplicates()     .rename(columns={'GeoAreaCode': 'id', 'GeoAreaName': 'name'})


# In[17]:


entities.to_csv('./exported_data/entities.csv', index=False)


# ## Export datasets and variables
# 
# Algorithm outline:
# 
#   - For each `INDICATOR`:
#     - Obtain dimensions (columns named `[between brackets]`) that contain non-null values
#       - For each combination of unique values values in those dimensions
#         - Generate a table of values.
# 

# In[18]:


DIMENSIONS = [c for c in expanded_df.columns if c[0] == '[' and c[-1] == ']']
NON_DIMENSIONS = [c for c in expanded_df.columns if c not in set(DIMENSIONS)]

@functools.lru_cache(maxsize=256)
def get_series_with_relevant_dimensions(indicator, series):
    """ For a given indicator and series, return a tuple:
    
      - data filtered to that indicator and series
      - names of relevant dimensions
      - unique values for each relevant dimension
    """
    data_filtered = expanded_df[(expanded_df.Indicator == indicator) & (expanded_df.SeriesCode == series)]
    non_null_dimensions_columns = [col for col in DIMENSIONS if data_filtered.loc[:, col].notna().any()]
    dimension_names = []
    dimension_unique_values = []
    
    for c in non_null_dimensions_columns:
        uniques = data_filtered[c].unique()
        if len(uniques) > 1:
            dimension_names.append(c)
            dimension_unique_values.append(list(uniques))

    return (data_filtered[NON_DIMENSIONS + dimension_names], dimension_names, dimension_unique_values)


# Generate tables for:
# 
#   - Rows where the dimension is `None`
#   - One table for each combination of unique values of relevant dimensions

# In[19]:


@functools.lru_cache(maxsize=256)
def generate_tables_for_indicator_and_series(indicator, series):
    tables_by_combination = {}
    data_filtered, dimensions, dimension_values = get_series_with_relevant_dimensions(indicator, series)
    if len(dimensions) == 0:
        # no additional dimensions
        export = data_filtered
        return export
    else:
        for dimension_value_combination in itertools.product(*dimension_values):
            # build filter by reducing, start with a constant True boolean array
            filt = [True] * len(data_filtered)
            for dim_idx, dim_value in enumerate(dimension_value_combination):
                dimension_name = dimensions[dim_idx]
                value_is_nan = type(dim_value) == float and math.isnan(dim_value)
                filt = filt                        & (data_filtered[dimension_name].isnull() if value_is_nan else data_filtered[dimension_name] == dim_value)

            tables_by_combination[dimension_value_combination] = data_filtered[filt].drop(dimensions, axis=1)
            
        return tables_by_combination
    


# In[20]:


all_series = expanded_df[['Indicator', 'SeriesCode', 'SeriesDescription', 'Units']]   .groupby(by=['Indicator', 'SeriesCode', 'SeriesDescription', 'Units'])   .count()   .reset_index()


# ### Export data
# 
# For each series and combination of additional dimensions' members, generate an entry in the `variables` table.

# In[ ]:


datasets = pd.DataFrame(columns=['id', 'name'])
variables = pd.DataFrame(columns=['id', 'name', 'unit', 'dataset_id'])
sources = pd.DataFrame(columns=['id', 'name', 'description', 'dataset_id'])

source_description_template = {
    'dataPublishedBy': "United Nations Statistics Division",
    'dataPublisherSource': None,
    'link': "https://unstats.un.org/sdgs/indicators/database/",
    'retrievedDate': datetime.now().strftime("%d-%B-%y"),
    'additionalInfo': None
}

variable_idx = 0

def extract_datapoints(df):
    return pd.DataFrame({
        'entity': df['GeoAreaCode'],
        'year': df['TimePeriod'],
        'value': df['Value']
    }).drop_duplicates(subset=['entity', 'year']).dropna()

for i, row in tqdm(all_series.iterrows(), total=len(all_series)):
    
    # DATASET
    
    datasets = datasets.append(
        {
            'id': i, 
            'name': row['SeriesDescription']
        }, 
        ignore_index=True)
    

    # SOURCE
    
    source_description = source_description_template.copy()
    
    try:
        source_description['additionalInfo'] = extract_description('metadata/Metadata-%s.pdf' % '-'.join([part.rjust(2, '0') for part in row['Indicator'].split('.')]))
    except:
        pass
    
    sources = sources.append({
        'id': i,
        'name': "%s (UN SDG, 2019)" % row['SeriesDescription'],
        'description': json.dumps(source_description),
        'dataset_id': i
    }, ignore_index=True)
        

    # VARIABLES & DATAPOINTS
    
    _, dimensions, dimension_members = get_series_with_relevant_dimensions(row['Indicator'], row['SeriesCode'])
    
    if len(dimensions) == 0:
        # no additional dimensions
        table = generate_tables_for_indicator_and_series(row['Indicator'], row['SeriesCode'])
        variable = {
            'id': variable_idx,
            'dataset_id': i,
            'unit': row['Units'],
            'name': "%s - %s - %s" % (row['Indicator'], row['SeriesDescription'], row['SeriesCode'])
        }
        variables = variables.append(variable, ignore_index=True)
        extract_datapoints(table).to_csv('./exported_data/%04d_datapoints.csv' % variable_idx, index=False)
        variable_idx += 1

    else:
        # has additional dimensions
        for member_combination, table in generate_tables_for_indicator_and_series(row['Indicator'], row['SeriesCode']).items():
            variable = {
                'id': variable_idx,
                'dataset_id': i,
                'unit': row['Units'],
                'name': "%s - %s - %s - %s" % (
                    row['Indicator'], 
                    row['SeriesDescription'], 
                    row['SeriesCode'],
                    ' - '.join(map(str, member_combination)))
                
            }
            variables = variables.append(variable, ignore_index=True)
            extract_datapoints(table).to_csv('./exported_data/%04d_datapoints.csv' % variable_idx, index=False)
            variable_idx += 1


variables.to_csv('./exported_data/variables.csv', index=False)
datasets.to_csv('./exported_data/datasets.csv', index=False)
sources.to_csv('./exported_data/sources.csv', index=False)


# In[ ]:




