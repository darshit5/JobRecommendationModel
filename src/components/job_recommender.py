import re
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
stopw = set(stopwords.words('english'))
from pyresparser import ResumeParser
import os
from docx import Document


# Load dataset:
jd_df = pd.read_csv(r'C:\Users\darshit joshi\Desktop\Model\src\data\jd_structured_data.csv')


skills = ['python','statistics']


def ngrams(string, n=3):
    string = fix_text(string)  # fix text
    string = string.encode("ascii", errors="ignore").decode()  # remove non ascii chars
    string = string.lower()
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title()  # normalise case - capital at start of each word
    string = re.sub(' +', ' ', string).strip()  # get rid of multiple spaces and replace with a single
    string = ' ' + string + ' '  # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
tfidf = vectorizer.fit_transform(skills)

nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
jd_test = (jd_df['Processed_JD'].values.astype('U'))

def getNearestN(query):
    queryTFIDF_ = vectorizer.transform(query)
    distances, indices = nbrs.kneighbors(queryTFIDF_)
    return distances, indices

distances, indices = getNearestN(jd_test)
test = list(jd_test)
matches = []

for i, j in enumerate(indices):
    dist = round(distances[i][0], 2)

    temp = [dist]
    matches.append(temp)

matches = pd.DataFrame(matches, columns=['Match confidence'])

jd_df['match']=matches['Match confidence']
# jd_df.head(5).sort_values('match')
jd_df.sort_values('match', ascending=False).head(5)
# Display recommended jobs using Streamlit
# st.title("Recommended Jobs")
# st.dataframe(jd_df.loc[indices.flatten()[:5]].sort_values('match')[['Job Title', 'Company Name', 'Location', 'Industry', 'Sector', 'Average Salary']])
