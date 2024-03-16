import streamlit as st
import pandas as pd
import altair as alt
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the data
df = pd.read_csv('sentiment_result.csv')

# Load the trained SVM and Logistic Regression models
with open('svm_model.pkl', 'rb') as svm_model_file:
    svm_model = pickle.load(svm_model_file)

with open('lr_model.pkl', 'rb') as lr_model_file:
    lr_model = pickle.load(lr_model_file)

# Load the CountVectorizer
with open('vectorizer.pkl', 'rb') as vector_file:
    vectorizer = pickle.load(vector_file)


@st.cache(allow_output_mutation=True)
def load_models():
    return svm_model, lr_model, vectorizer


svm_model, lr_model, vectorizer = load_models()


def plot_pie_chart(avg_positive, avg_negative, avg_neutral):
    data = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Count': [avg_positive, avg_negative, avg_neutral]
    })

    chart = alt.Chart(data).mark_bar().encode(
        x='Sentiment',
        y='Count',
        color='Sentiment'
    ).properties(
        title='Average Sentiment Analysis',
        width=400,
        height=300
    )

    st.altair_chart(chart)


def perform_sentiment_analysis(college_df, keyword):
    # Filter the dataset based on the keyword
    keyword_df = college_df[college_df['Review'].str.contains(keyword, case=False, regex=True)]

    # Vectorize the reviews using the loaded vectorizer
    reviews_vectorized = vectorizer.transform(keyword_df['Review'])

    # Perform sentiment analysis using the loaded model
    svm_sentiments = svm_model.predict(reviews_vectorized)
    lr_sentiments = lr_model.predict(reviews_vectorized)

    # Combine sentiments from both models
    combined_sentiments = [f"{svm}_{lr}" for svm, lr in zip(svm_sentiments, lr_sentiments)]

    positive_count = sum(1 for score in combined_sentiments if score == 'Positive_Positive')
    negative_count = sum(1 for score in combined_sentiments if score == 'Negative_Negative')
    neutral_count = sum(1 for score in combined_sentiments if score == 'Neutral_Neutral')

    total_reviews = len(combined_sentiments)
    avg_positive = positive_count / total_reviews
    avg_negative = negative_count / total_reviews
    avg_neutral = neutral_count / total_reviews

    return positive_count, negative_count, neutral_count, avg_positive, avg_negative, avg_neutral


def main():
    st.title("Sentiment Analysis Web App")

    page = st.sidebar.selectbox("Select a page", ["Home", "Overview", "Team", "Contact", "Feedback", "Result"])

    if page == "Home":
        st.header("Home Page")
        home_html = """
           <!DOCTYPE html>
           <html lang="en">
           <head>
               <meta charset="UTF-8">
               <meta name="viewport" content="width=device-width, initial-scale=1.0">
               <title>Engineering College Sentiment Analysis</title>
               <!-- Add Bootstrap CDN link -->
               <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous">
               <!-- Add custom styles -->
               <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
               <style>
                   body {
                       background: url("static/sharp_purple.png") center center fixed;
                       background-size: cover;
                       background-repeat: no-repeat;
                       background-color: #f8f9fa;
                       display: flex;
                       flex-direction: column;
                       min-height: 100vh;
                       margin-bottom: 100px; /* Adjust margin for footer */
                       color: #343a40; /* Set text color */
                   }

                   .container {
                       max-width: 600px;
                       margin: auto;
                       background-color: #fff;
                       padding: 30px;
                       border-radius: 8px;
                       box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                       flex: 1; /* Fill available space */
                   }

                   h1,
                   .display-4 {
                       font-size: 2.5rem; /* Adjust the font size as needed */
                       text-align: center; /* Center-align the text */
                       margin-bottom: 20px; /* Add spacing below the heading */
                   }

                   .welcome-message {
                       font-size: 2rem;
                       text-align: center;
                       margin-bottom: 10px;
                       color: #007bff; /* Set welcome message color */
                   }

                   label {
                       font-weight: bold;
                   }

                   button {
                       background-color: #007bff;
                       color: #fff;
                       border: none;
                       padding: 10px 20px;
                       border-radius: 4px;
                       cursor: pointer;
                   }

                   button:hover {
                       background-color: #0056b3;
                   }

                   /* Add styles for the footer */
                   footer {
                       position: fixed;
                       bottom: 0;
                       background-color: #343a40;
                       color: #304146; /* Lighter text color */
                       padding: 20px 0;
                       width: 100%;
                   }

                   footer p {
                       margin-bottom: 0;
                   }

                   footer a {
                       color: #dee2e6; /* Lighter link color */
                   }

                   footer a:hover {
                       text-decoration: underline;
                   }

                   /* Adjust icon size */
                   footer img {
                       width: 30px; /* Adjust the width as needed */
                       height: auto; /* Maintain the aspect ratio */
                       margin-right: 10px; /* Add some spacing between icons */
                   }

                   /* Style for logout link */
                   .logout-link {
                       text-align: center;
                       margin-top: 10px;
                   }

                   .logout-link a {
                       color: #dc3545; /* Set logout link color */
                       text-decoration: none;
                   }

                   .logout-link a:hover {
                       text-decoration: underline;
                   }
               </style>
           </head>
           <body>
               <div class="container text-center">
                   <h1 class="mb-4 display-4">Select a College and Enter a Keyword</h1>
                   <form action="/result" method="post">
                       <div class="mb-3">
                           <label for="college" class="form-label">Select a College:</label>
                           <select name="college" id="college" class="form-select" required>
                               {% for college in colleges %}
                                   <option value="{{ college }}">{{ college }}</option>
                               {% endfor %}
                           </select>
                       </div>
                       <div class="mb-3">
                           <label for="keyword" class="form-label">Enter a Keyword:</label>
                           <input type="text" name="keyword" id="keyword" class="form-control" required>
                       </div>
                       <button type="submit">Submit</button>
                   </form>
               </div>
               <!-- Footer -->
               <footer class="fixed-bottom">
                   <div class="container text-center">
                       <p>&copy; 2023 Engineering College Sentiment Analysis</p>
                       <p>Follow us on social media:
                           <a href="https://twitter.com" class="me-2"><img src="{{ url_for('static', filename='twitter.png') }}" alt="Twitter"></a>
                           <a href="https://www.facebook.com" class="me-2"><img src="{{ url_for('static', filename='facebook.png') }}" alt="Facebook"></a>
                           <a href="https://www.linkedin.com" class="me-2"><img src="{{ url_for('static', filename='linkedin.png') }}" alt="LinkedIn"></a>
                           <a href="https://www.instagram.com/anpatil2621" class="me-2"><img src="{{ url_for('static', filename='instagram.png') }}" alt="Instagram"></a>
                           <a href="https://github.com/Anpatil1" class="me-2"><img src="{{ url_for('static', filename='github.png') }}" alt="GitHub"></a>
                       </p>
                   </div>
               </footer>

               <!-- Add Bootstrap JS and Popper.js CDN scripts -->
               <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-eA7mIjNjha5tOOt66G0RJuFbp5K8JjIbOvuS6uqUwn0/pFfYFVxj4E1bISJKDL/S" crossorigin="anonymous"></script>
               <!-- Add Font Awesome CDN for social media icons -->
               <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.2/js/all.min.js" integrity="sha384-lBZdsY2jPFO9ZW9KOoPZhu1LxH7ZpH0GqtczVp9i7uR8cbrnw2WhF0NOhjsJ2OV9" crossorigin="anonymous"></script>
           </body>
           <!-- Your script and style links here -->
           </html>
           """
        st.markdown(home_html, unsafe_allow_html=True)

    elif page == "Overview":
        st.header("Overview Page")
        st.markdown("<h1>Overview Page</h1>", unsafe_allow_html=True)
        # Add content for the overview page

    elif page == "Team":
        st.header("Team Page")
        st.markdown("<h1>Team Page</h1>", unsafe_allow_html=True)
        # Add content for the team page

    elif page == "Contact":
        st.header("Contact Page")
        st.markdown("<h1>Contact Page</h1>", unsafe_allow_html=True)
        # Add content for the contact page

    elif page == "Feedback":
        st.header("Feedback Page")
        st.markdown("<h1>Feedback Page</h1>", unsafe_allow_html=True)
        # Add content for the feedback page

    elif page == "Result":
        st.header("Result Page")
        st.markdown("<h1>Result Page</h1>", unsafe_allow_html=True)
        # Add content for the result page

        # Example: Perform sentiment analysis and display results
        college_name = st.selectbox("Select College", sorted(df['college_name'].unique()))
        keyword = st.text_input("Enter Keyword", "")

        if st.button("Submit"):
            college_df = df[df['college_name'] == college_name]
            positive_count, negative_count, neutral_count, avg_positive, avg_negative, avg_neutral = perform_sentiment_analysis(
                college_df, keyword)
            plot_pie_chart(avg_positive, avg_negative, avg_neutral)

            review_count = len(college_df)

            st.subheader("Analysis Results")
            st.write(f"College: {college_name}")
            st.write(f"Keyword: {keyword}")
            st.write(f"Total Reviews: {review_count}")
            st.write(f"Positive Count: {positive_count}")
            st.write(f"Negative Count: {negative_count}")
            st.write(f"Neutral Count: {neutral_count}")
            st.write(f"Avg Positive: {avg_positive:.2%}")
            st.write(f"Avg Negative: {avg_negative:.2%}")
            st.write(f"Avg Neutral: {avg_neutral:.2%}")


if __name__ == '__main__':
    main()
