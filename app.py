from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import pandas as pd
from flask import Flask, render_template, jsonify
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'An@2621#'

# Connect to SQLite database
db = sqlite3.connect('flask_users.db', check_same_thread=False)
cursor = db.cursor()

# Create users table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        password TEXT NOT NULL,
        email TEXT NOT NULL
    )
''')

db.commit()


df = pd.read_csv('engineering_colleges_reviews.csv')

with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vector_file:
    vectorizer = pickle.load(vector_file)

plt.switch_backend('agg')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if request.method == 'POST':
        college = request.form.get('college')
        keyword = request.form.get('keyword')
        name = request.form.get('name')
        email = request.form.get('email')
        rating = request.form.get('rating')
        comments = request.form.get('comments')

        # Save the feedback to the SQLite database
        save_feedback_to_database(college, keyword, name, email, rating, comments)

        # Return JSON response
        # Return a JSON response
        return jsonify({'title': 'Feedback Submitted', 'text': 'Thank you for your feedback!', 'icon': 'success'})

# Add the function to save feedback to the SQLite database
def save_feedback_to_database(college, keyword, name, email, rating, comments):
    # Connect to the database (replace 'feedback.db' with your desired database name)
    connection = sqlite3.connect('feedback.db')
    cursor = connection.cursor()

    # Create a table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            college TEXT,
            keyword TEXT,
            name TEXT,
            email TEXT,
            rating INTEGER,
            comments TEXT
        )
    ''')

    # Insert feedback into the database
    cursor.execute('''
        INSERT INTO feedback (college, keyword, name, email, rating, comments)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (college, keyword, name, email, rating, comments))

    # Commit the changes and close the connection
    connection.commit()
    connection.close()




def plot_pie_chart(avg_positive, avg_negative, avg_neutral):
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [avg_positive, avg_negative, avg_neutral]
    colors = ['#77DD77', '#FF6961', '#AEC6CF']
    explode = (0.1, 0, 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=140, explode=explode, wedgeprops=dict(width=0.4)
    )

    plt.setp(autotexts, size=8, weight="bold")

    ax.set_title('Average Sentiment Analysis')

    img_stream = BytesIO()
    fig.savefig(img_stream, format='png')
    img_stream.seek(0)
    img_data = base64.b64encode(img_stream.read()).decode('utf-8')

    plt.close(fig)
    return img_data


import numpy as np

def perform_sentiment_analysis(college_df, keyword):
    # Filter the dataset based on the keyword
    keyword_df = college_df[college_df['Review'].str.contains(keyword, case=False, regex=True)]

    # Vectorize the reviews using the loaded vectorizer
    reviews_vectorized = vectorizer.transform(keyword_df['Review'])

    # Perform sentiment analysis using the loaded model
    sentiment_scores = model.predict(reviews_vectorized)

    if len(sentiment_scores) == 0:
        # Handle the case where there are no reviews with the specified keyword
        return 0, 0, 0, 0, 0, 0

    positive_count = sum(1 for score in sentiment_scores if score == 'Positive')
    negative_count = sum(1 for score in sentiment_scores if score == 'Negative')
    neutral_count = sum(1 for score in sentiment_scores if score == 'Neutral')

    total_reviews = len(sentiment_scores)
    avg_positive = positive_count / total_reviews
    avg_negative = negative_count / total_reviews
    avg_neutral = neutral_count / total_reviews

    return positive_count, negative_count, neutral_count, avg_positive, avg_negative, avg_neutral

@app.route('/')
def index():
    colleges = sorted(df['college_name'].unique())
    return render_template('index.html', colleges=colleges)

@app.route('/overview')
def overview():
    # Your overview route logic...
    return render_template('overview.html')

@app.route('/team')
def team():

    return render_template('team.html')

@app.route('/contact')
def contact():

    return render_template('contact.html')


@app.route('/feedback')
def feedback():
    # You can pass any required data to the template
    college = "Your College"
    keyword = "Your Keyword"
    return render_template('feedback.html', college=college, keyword=keyword)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'username' in session:
        flash('You are already logged in.', 'info')
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        email = request.form['email']

        # Check if the passwords match
        if password != confirm_password:
            flash('Passwords do not match. Please enter matching passwords.', 'error')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='sha256')

        cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Username or email already exists. Choose different ones.', 'error')
        else:
            cursor.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                           (username, hashed_password, email))
            db.commit()
            flash('Account created successfully! You can now log in.', 'success')
            return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        flash('You are already logged in.', 'info')
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()

        if user and check_password_hash(user[2], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'error')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if 'username' in session:
        flash('You are already logged in.', 'info')
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form['email']

        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()

        if user:
            # Generate a temporary token
            token = secrets.token_hex(16)

            # Save the token in the database
            cursor.execute("UPDATE users SET reset_token = ? WHERE email = ?", (token, email))
            db.commit()

            # Send email with reset link
            reset_link = url_for('reset_password', token=token, _external=True)
            msg = Message('Password Reset Request', recipients=[email])
            msg.body = f'To reset your password, click the following link: {reset_link}'
            mail.send(msg)

            flash('Password reset instructions sent to your email.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Invalid email. Please try again.', 'error')

    return render_template('Forgot Password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if 'username' in session:
        flash('You are already logged in.', 'info')
        return redirect(url_for('index'))

    cursor.execute("SELECT * FROM users WHERE reset_token = ?", (token,))
    user = cursor.fetchone()

    if not user:
        flash('Invalid or expired reset token.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match. Please enter matching passwords.', 'error')
            return redirect(url_for('reset_password', token=token))

        hashed_password = generate_password_hash(password, method='sha256')

        # Update the user's password and clear the reset token
        cursor.execute("UPDATE users SET password = ?, reset_token = NULL WHERE id = ?", (hashed_password, user[0]))
        db.commit()

        flash('Password reset successful. You can now log in with your new password.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)



@app.route('/result', methods=['GET', 'POST'])
def result():
    if 'username' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        college_name = request.form['college']
        keyword = request.form['keyword']

        college_df = df[df['college_name'] == college_name]
        positive_count, negative_count, neutral_count, avg_positive, avg_negative, avg_neutral = perform_sentiment_analysis(
            college_df, keyword)

        img_data = plot_pie_chart(avg_positive, avg_negative, avg_neutral)

        review_count = len(college_df)

        return render_template('result'
                               '.html', college=college_name, keyword=keyword, img_data=img_data,
                               review_count=review_count, positive_count=positive_count,
                               negative_count=negative_count, neutral_count=neutral_count,
                               avg_positive=avg_positive, avg_negative=avg_negative, avg_neutral=avg_neutral)

    colleges = sorted(df['college_name'].unique())
    return render_template('result.html', colleges=colleges)



if __name__ == '__main__':
    app.run(threaded=True, debug=True)
