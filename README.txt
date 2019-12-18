============================== HOW TO RUN =============================

This will detail how to run the system on a linux-based machine. If
using Windows, please use a git-bash terminal. 

1. Ensure that all the correct imports are installed in the environment,
namely flask, numpy, pandas, scipy.sparse.linalg. All these imports
can also be found at the top of bookrecc.py. Python 3.7.1 was used in
development. 

2. From inside the main directory, run the following command
        $python3 bookrecc.py
   
   You will expect an output similar to this:
        * Serving Flask app "bookrecc" (lazy loading)
        * Environment: production
          WARNING: This is a development server. Do not use it in a 
	  production deployment.
          Use a production WSGI server instead.
        * Debug mode: on
        * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    
    Follow the specified IP address to access the single-page web app.
    The initial start up might take a few seconds as the .cvs files are
    loaded into the server. The initial "Get my recommendations" request
    might also take a few seconds. The GET and POST requests are logged 
    in the terminal from which the python script is run. 


=========================== ABOUT THE SYSTEM ==========================

The Flask server reads in 2 .cvs files: 
1. data_books.cvs - 4 columns structured as follows: 
    (Book_ID, Authors, Title, Image)
    10,000 entries
2.data_ratings.cvs - 3 columns structured as follows:
    (User_ID, Book_ID, Rating)
    100,000 entries
Both of these datasets were obtained from goodbooks-10k:
https://github.com/zygmuntz/goodbooks-10k (02.12.2019)

The system generates recommendations using Matrix Factorization via 
Singular Value Decomposition. The following was used as reference:
https://beckernick.github.io/matrix-factorization-recommender/

Ratings can be added by selecting "Make some ratings", where a random 
selection of 20 unrated books will present itself. The user can select
this again to refresh the options. Each rating is an individual form
and will be recorded accordingly.Each time that the user adds a rating, 
the global dataframe is updated.The user can choose to rate any number 
of the books presented. The effects can immediately be seen by clicking 
"Get my recommendations".Each time that the user selects "Get my 
recommendations", the system will recalculate the recommendations based 
on the most up-to-date dataframe available. 

The user is defaulted to user 1. One can choose any other user profile
by changing the variable 'user' in bookrecc.py.

*note: some of the books have foreign characters that were not preserved 
in their names/Authors and thus appear strangely on the webapp. 