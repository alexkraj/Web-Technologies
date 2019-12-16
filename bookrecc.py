# CTVT58
# SSA - Web Technologies
# Recommendation system
# Dataset from GoodBooks

from flask import Flask, render_template  # , request, redirect, response
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np

# import sys
# import random
# import json

# max user_id in dataset is 24420
user = 24421
app = Flask(__name__)

# read the BOOKS cvs and convert into panda dataframe
data_books = [
    i.strip().split(";")
    for i in open("data/data_books.csv", "r").readlines()
]
books_df = pd.DataFrame(
    data_books, columns=["Book_ID", "Authors", "Title", "Image"]
)
books_df['Book_ID'] = books_df['Book_ID'].apply(pd.to_numeric, errors='coerce')

# read the RATINGS cvs and convert into panda dataframe
data_ratings = [
    i.strip().split(";")
    for i in open("data/data_ratings.csv", "r").readlines()
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

# making the recommendations matrix, fill the rest with 0
R_df = ratings_df.pivot(index="User_ID",
                        columns="Book_ID",
                        values="Rating").fillna(0)

# print(R_df.head())

# de-meaning the data
# R = R_df.as_matrix()
R = R_df.rename_axis('ID').values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# singular value decomposition
U, sigma, Vt = svds(R_demeaned, k=50)
sigma = np.diag(sigma)

# making predictions from decomposed matrices
all_user_predicted_ratings = (np.dot(np.dot(U, sigma), Vt) +
                              user_ratings_mean.reshape(-1, 1))
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns)


def recommend_movies(predictions_df, userID, books_df, original_ratings_df, num_recommendations=15):
    user_row_number = userID
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    user_data = original_ratings_df[original_ratings_df.User_ID == (userID)]
    user_full = (user_data.merge(books_df,
                                 how='left',
                                 left_on='Book_ID',
                                 right_on='Book_ID'
                                 ).sort_values(['Rating'], ascending=False))
    print('User {0} has already rated {1} books.'.format(userID, user_full.shape[0]))
    print('Recommending the highest {0} predicted ratings books not already rated.'.format(num_recommendations))

    recommendations = (books_df[~books_df['Book_ID'].isin(user_full['Book_ID'])].
                       merge(pd.DataFrame(sorted_user_predictions).reset_index(),
                             how='left',
                             left_on='Book_ID',
                             right_on='Book_ID').rename(
                                columns={user_row_number: 'Predictions'}).sort_values(
                                    'Predictions', ascending=False).iloc[:num_recommendations, :-1]
                       )
    return user_full, recommendations


already_rated, predictions = recommend_movies(preds_df, 4, books_df, ratings_df, 15)
print(already_rated.head(15))

@app.route("/")
def main():
    return render_template("home.html")


# if __name__ == "__main__":
#     app.run(debug=True)
