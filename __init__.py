import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.components.job_recommender import ngrams, getNearestN, jd_df
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to process the entered skills and recommend jobs
def process_skills(skills):
    # Feature Engineering:
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
    tfidf = vectorizer.fit_transform([skills])

    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
    jd_test = (jd_df['Processed_JD'].values.astype('U'))

    # distances, indices = getNearestN(jd_test)
    queryTFIDF_ = vectorizer.transform(jd_test)
    distances, indices = nbrs.kneighbors(queryTFIDF_)
    test = list(jd_test)
    matches = []

    for i, j in enumerate(indices):
        dist = round(distances[i][0], 2)
        temp = [dist]
        matches.append(temp)

    matches = pd.DataFrame(matches, columns=['Match confidence'])

    # Following recommends Top 5 Jobs based on entered skills:
    jd_df['match'] = matches['Match confidence']

    return jd_df.sort_values('match').head(5)

# Streamlit app
def main():
    st.title("Job Recommendation Model")
    st.write("Enter your skills in the text box below:")

    # Text box for entering skills
    
    skills = st.text_area("Enter your skills here:")
    
    if st.button("Recommend Jobs") and skills is not None:
        # Process entered skills and recommend jobs
        df_jobs = process_skills(skills)

        # Display recommended jobs as DataFrame
        st.write("Recommended Jobs:")
        st.dataframe(df_jobs[['Job Title', 'Company Name', 'Location', 'Industry', 'Sector', 'Average Salary']])

# Run the Streamlit app
if __name__ == '__main__':
    main()
