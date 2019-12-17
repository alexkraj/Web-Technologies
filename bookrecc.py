# CTVT58
# SSA - Web Technologies
# Recommendation system
# Dataset from GoodBooks
from flask import Flask, render_template, request
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np
import random
import json

app = Flask(__name__)
user = 1
user_books = []  # stores book IDs that the user has already rated

books_df = []
ratings_df = []


# read the BOOKS and RATINGS and convert into panda dataframe
def read_data(d_books, d_ratings):
    data_books = [
        i.strip().split(";")
        for i in open(d_books, "r").readlines()
    ]
    books_df = pd.DataFrame(
        data_books, columns=["Book_ID", "Authors", "Title", "Image"]
    )
    books_df['Book_ID'] = books_df['Book_ID'].apply(pd.to_numeric,
                                                    errors='coerce')

    data_ratings = [
        i.strip().split(";")
        for i in open(d_ratings, "r").readlines()
    ]
    ratings_df = pd.DataFrame(
        data_ratings, columns=["User_ID", "Book_ID", "Rating"], dtype=int
    )
    ratings_df['User_ID'] = ratings_df['User_ID'].apply(pd.to_numeric,
                                                        errors='coerce')
    ratings_df['Book_ID'] = ratings_df['Book_ID'].apply(pd.to_numeric,
                                                        errors='coerce')
    ratings_df['Rating'] = ratings_df['Rating'].apply(pd.to_numeric,
                                                      errors='coerce')
    return books_df, ratings_df


# making the recommendations matrix, fill the rest with 0s
def create_matrix(ratings):
    recommendation_matrix = ratings.pivot(index="User_ID",
                                          columns="Book_ID",
                                          values="Rating").fillna(0)
    return recommendation_matrix


# demeaning the data
def demean_data(ratings):
    R = ratings.rename_axis('ID').values
    user_ratings_mean = np.mean(R, axis=1)
    demeaned_ratings = R - user_ratings_mean.reshape(-1, 1)
    return demeaned_ratings, user_ratings_mean


def recommend_books(predictions_df, userID, books_df,
                    original_ratings_df, num_recommendations=15):
    user_row_number = userID
    sorted_user_predictions = (predictions_df.
                               iloc[user_row_number].
                               sort_values(ascending=False))

    user_data = original_ratings_df[original_ratings_df.User_ID == (userID)]
    user_full = (user_data.merge(books_df,
                                 how='left',
                                 left_on='Book_ID',
                                 right_on='Book_ID'
                                 ).sort_values(['Rating'], ascending=False))
    print('User {0} has already rated {1} books.'.format(userID,
                                                         user_full.shape[0]))
    print('''Recommending the highest {0} predicted ratings books not already rated.'''.format(num_recommendations))

    recommendations = (books_df[~books_df['Book_ID'].isin(user_full['Book_ID'])].
                       merge(pd.DataFrame(sorted_user_predictions).reset_index(),
                             how='left',
                             left_on='Book_ID',
                             right_on='Book_ID').rename(
                                columns={user_row_number: 'Predictions'}).sort_values(
                                    'Predictions', ascending=False).iloc[:num_recommendations, :-1]
                       )
    return user_full, recommendations


@app.route("/")
def home():
    message = ""
    user_info = {'user_ID': user,
                 'message': message}
    return render_template("home.html", user=user_info)


@app.route("/myrecc", methods=['GET'])
def my_recc():

    R_df = create_matrix(ratings_df)
    R_demeaned, user_ratings_mean = demean_data(R_df)

    # singular value decomposition
    U, sigma, Vt = svds(R_demeaned, k=50)
    sigma = np.diag(sigma)

    # making predictions from decomposed matrices
    all_user_predicted_ratings = (np.dot(np.dot(U, sigma), Vt) +
                                  user_ratings_mean.reshape(-1, 1))
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns)

    already_rated, predictions = recommend_books(preds_df,
                                                 user,
                                                 books_df,
                                                 ratings_df,
                                                 10)

    recommended_books = already_rated.head(10)
    books_json = recommended_books.to_json(orient="records")
    user_info = {'user_ID': user,
                 'rec_books': books_json}

    return json.dumps(user_info)


@app.route("/rate", methods=['GET'])
def rate():
    book_ids = []
    for i in range(0, 15):
        val = random.randint(1, 10000)
        if((val in book_ids) or (val in user_books)):
            break
        book_ids.append(val)

    book_names = []
    book_images = []
    book_authors = []

    books_df.set_index("Book_ID")
    for book_id in book_ids:
        book_names.append(books_df.loc[book_id]["Title"])
        book_images.append(books_df.loc[book_id]["Image"])
        book_authors.append(books_df.loc[book_id]["Authors"])

    books = {'book_IDs': book_ids,
             'book_titles': book_names,
             'book_images': book_images,
             'book_authors': book_authors}
    return json.dumps(books)


@app.route("/addRating", methods=['POST'])
def addRating():
    rating = int(request.form['rating'])
    book = int(request.form['book_id'])
    user_books.append(book)

    arr = ["User_ID", "Book_ID", "Rating"]
    temp = pd.DataFrame([[user, book, rating]], columns=arr)

    global ratings_df

    ratings_df_new = ratings_df.append(temp, ignore_index=True)
    ratings_df = ratings_df_new
    return "success"


if __name__ == "__main__":
    books_df, ratings_df = read_data("data/data_books.csv", "data/data_ratings.csv")
    app.run(debug=True)
