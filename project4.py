#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np


# In[24]:


def get_movies():
    git_url = "https://raw.githubusercontent.com/jungmc2/movie_recommender/main/"
    
    movies = pd.read_csv(git_url + 'movies.dat', sep='::', engine = 'python',
                     encoding="ISO-8859-1", header = None)
    movies.columns = ['MovieID', 'Title', 'Genres']
    
    small_image_url = "https://liangfgithub.github.io/MovieImages/"
    movies['image_url'] = movies.MovieID.apply(lambda x: small_image_url + str(x) + ".jpg")
    return movies

def get_system_data():
    git_url = "https://raw.githubusercontent.com/jungmc2/movie_recommender/main/"
    return pd.read_csv(git_url + 'system2_df.csv', index_col=0)

def get_similarity():
    git_url = "https://raw.githubusercontent.com/jungmc2/movie_recommender/main/"
    return pd.read_csv(git_url + 's_matrix.csv', index_col = 0)


# In[25]:


genres = ['Action',
 'Adventure',
 'Animation',
 "Children's",
 'Comedy',
 'Crime',
 'Documentary',
 'Drama',
 'Fantasy',
 'Film-Noir',
 'Horror',
 'Musical',
 'Mystery',
 'Romance',
 'Sci-Fi',
 'Thriller',
 'War',
 'Western']

git_url = "https://raw.githubusercontent.com/jungmc2/movie_recommender/main/"


# In[26]:


def myIBCF(newuser):
    s = get_similarity()
    predictions = np.zeros(len(newuser))
    
    unrated_movies = np.where(newuser.isna())[0]
    for i in unrated_movies:
        neighbors = np.argsort(s.iloc[i].fillna(0).values)[::-1][:30]

        denom_mask = [i for i in neighbors if i not in unrated_movies]

        numerator = np.sum(s.iloc[i, neighbors] * newuser.iloc[neighbors])
        denominator = np.sum(s.iloc[i, denom_mask])

        if denominator != 0:
            predictions[i] = numerator/denominator 

    rec_df = pd.DataFrame()
    rec_df['movie_id'] = s.iloc[np.argsort(predictions)][::-1].index
    rec_df['rating'] = np.sort(predictions)[::-1]
    
    return rec_df.iloc[:10,]


# In[27]:


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(html.H1("Movie Recommender App", className="text-center mb-4")),
            className="mb-4",
        ),
        dcc.Tabs(
            [
                dcc.Tab(
                    label='Recommendation Based on Genres',
                    children=[
                        html.Div(
                            [
                                html.P('Step 1: Please select your favorite genre'),
                                dcc.Dropdown(
                                    id='dropdown-menu',
                                    options=[{'label': option, 'value': option} for option in genres],
                                    value=genres[0],
                                    multi=False,
                                    className="mb-4",
                                ),
                                html.P('Step 2: View movie recommendations'),
                                dbc.Button(
                                    'Click to display movie recommendations',
                                    id='display-button',
                                    n_clicks=0,
                                    color='primary',
                                    className="mb-4",
                                ),
                                html.Div(id='movie-list-container'),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label='Recommendation Based on User Ratings',
                    children=[
                        html.Div(
                            [
                                html.P('Please Rate the movies below '),
                                html.Div(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H3('Toy Story (1995)', style={'fontSize': '20px', 'textAlign': 'center'}),  # Adjust font size and alignment
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/1.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating1', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,  # Adjust the column width based on your preference
                                                ),
                                                # Repeat the structure for other images and columns
                                                
                                                dbc.Col(
                                                    [
                                                        html.H3('Grumpier Old Men (1995)', style={'fontSize': '20px', 'textAlign': 'center'}),
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/3.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating2', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,
                                                ),
                                                    dbc.Col(
                                                    [
                                                        html.H3('GoldenEye (1995)', style={'fontSize': '20px', 'textAlign': 'center'}),  # Adjust font size and alignment
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/10.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating3', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,  # Adjust the column width based on your preference
                                                ),
                                                # Repeat the structure for other images and columns
                                                
                                                dbc.Col(
                                                    [
                                                        html.H3('Cutthroat Island (1995)', style={'fontSize': '20px', 'textAlign': 'center'}),
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/15.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating4', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,
                                                )
                                                
                                                # Repeat the structure for other images and columns
                                            ],
                                            className="mb-4",
                                        ),
                                    
                                       dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H3('Four Rooms (1995)', style={'fontSize': '20px', 'textAlign': 'center'}),  # Adjust font size and alignment
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/18.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating5', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,  # Adjust the column width based on your preference
                                                ),
                                                # Repeat the structure for other images and columns
                                                
                                                dbc.Col(
                                                    [
                                                        html.H3('Now and Then (1995)', style={'fontSize': '20px', 'textAlign': 'center'}),
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/27.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating6', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,
                                                ),
                                                    dbc.Col(
                                                    [
                                                        html.H3('Sudden Death (1995)', style={'fontSize': '20px', 'textAlign': 'center'}),  # Adjust font size and alignment
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/9.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating7', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,  # Adjust the column width based on your preference
                                                ),
                                                # Repeat the structure for other images and columns
                                                
                                                dbc.Col(
                                                    [
                                                        html.H3('Get Shorty (1995)', style={'fontSize': '20px', 'textAlign': 'center'}),
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/21.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating8', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,
                                                )
                                                
                                                # Repeat the structure for other images and columns
                                            ],
                                            className="mb-4",
                                        ),                                        dbc.Button(
                                            'Submit Ratings',
                                            id='submit-ratings-button',
                                            n_clicks=0,
                                            color='primary',
                                            className="mb-4",
                                        ),
                                    ]
                                ),
                                html.Div(id='movie-list-container2'),
                            ]
                        ),
                        html.H3("Here are some movie recommendations"),
                        html.Div(id='movie-recommendations'),
                    ],
                ),
            ]
        ),
    ],
    className="p-5",
)

@app.callback(
    Output('movie-list-container', 'children'),
    [Input('display-button', 'n_clicks')],
    [State('dropdown-menu', 'value')]
)
def update_movie_list(n_clicks, selected_genre):
    if n_clicks > 0:
        selected_data = pd.read_csv(git_url + selected_genre + ".csv")
        movie_list = [
            html.Div(
                [
                    html.H3(title),
                    html.Img(src=image_url, style={'width': '200px', 'height': '200px'}),
                ],
                style={'margin-bottom': '20px'}  # Add margin to separate movie elements
            )
            for title, image_url in zip(selected_data['Title'], selected_data['image_url'])
        ]
        return movie_list
    return ""


@app.callback(
    Output('movie-list-container2', 'children'),
    [Input('submit-ratings-button', 'n_clicks')],
    [State(f'rating{i}', 'value') for i in range(1, 9)]
)
def update_ratings_list(n_clicks, *ratings):
    global user_ratings
    if n_clicks > 0:
        # Store the ratings in the global variable
        user_ratings = list(ratings)
        return html.P(f'Ratings submitted: {user_ratings}')
    return []

# Your function to use the combined data
def process_ratings_data(movie_data, user_ratings):
    combined_data = []

    # Assuming 'Title' is a key in your movie_data
    for title, rating in zip(to_rate, user_ratings):
        combined_data.append({'MovieID': title, 'Rating': rating})

    return combined_data



@app.callback(
    Output('movie-recommendations', 'children'),
    [Input('submit-ratings-button', 'n_clicks')],
    [State(f'rating{i}', 'value') for i in range(1, 9)]
)
def generate_movie_recommendations(n_clicks, *ratings):
    global user_ratings
    if n_clicks > 0:
        # Store the ratings in the global variable
        user_ratings = list(ratings)
        to_rate = ['m1','m3','m10','m15','m18','m27','m9','m21']
        combined = pd.DataFrame({'newuser':ratings}, index=to_rate)
        
        system2_df = get_system_data()
        newuser = system2_df.merge(combined, how='left', left_index=True, right_index=True)['newuser']
        predictions = myIBCF(newuser)
        
        movies2 = get_movies().loc[:,['MovieID','Title','image_url']]
        movies2['MovieID'] = movies2['MovieID'].apply(lambda x: 'm' + str(x))
        
        recommendations = movies2.merge(predictions, how = 'inner', left_on='MovieID',right_on ='movie_id')
        
        # Generate HTML for displaying recommendations
        movie_recommendations = [
            html.Div(
                [
                    html.H3(title, style={'fontSize': '20px'}),
                    html.Img(src=image_url, style={'width': '50%', 'height': '200px'}),
                ],
                style={'margin-bottom': '20px'}
            )
            for title, image_url in zip(recommendations['Title'], recommendations['image_url'])
        ]

        return movie_recommendations
    return []

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




