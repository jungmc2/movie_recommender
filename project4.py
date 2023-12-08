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
    
    # movies = pd.read_csv(git_url + 'movies.dat', sep='::', engine = 'python',
    #                  encoding="ISO-8859-1", header = None)
    # movies.columns = ['MovieID', 'Title', 'Genres']
    
    # small_image_url = "https://liangfgithub.github.io/MovieImages/"
    # movies['image_url'] = movies.MovieID.apply(lambda x: small_image_url + str(x) + ".jpg")
    movies = pd.read_csv(git_url + 'movies_subset_final.csv', index_col=0)
    return movies

def get_system_data():
    git_url = "https://raw.githubusercontent.com/jungmc2/movie_recommender/main/"
    return pd.read_csv(git_url + 'system2_subset_final.csv', index_col=0)

def get_similarity():
    git_url = "https://raw.githubusercontent.com/jungmc2/movie_recommender/main/"
    return pd.read_csv(git_url + 's_subset_final.csv', index_col = 0)


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
            dbc.Col(html.H1("Movie Recommender App", className="text-center object")),
            className="object",
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
                                    id='Genre Selection',
                                    options=[{'label': option, 'value': option} for option in genres],
                                    value=genres[0],
                                    multi=False,
                                    className="object",
                                ),
                                html.P('Step 2: View movie recommendations'),
                                dbc.Button(
                                    'Click to display movie recommendations',
                                    id='Display Movies',
                                    n_clicks=0,
                                    color='primary',
                                    className="object",
                                ),
                                html.Div(id='Movie Recommendations'),
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
                                                        html.H3('Tom and Huck (1995)', style={'fontSize': '20px', 'textAlign': 'center'}),  # Adjust font size and alignment
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/8.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating1', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,  
                                                ),
                                                
                                                dbc.Col(
                                                    [
                                                        html.H3('Across the Sea of Time (1995)', style={'fontSize': '20px', 'textAlign': 'center'}),
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/37.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating2', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,
                                                ),
                                                    dbc.Col(
                                                    [
                                                        html.H3('It Takes Two (1995)', style={'fontSize': '20px', 'textAlign': 'center'}),  # Adjust font size and alignment
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/38.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating3', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,  
                                                ),
                                                
                                                dbc.Col(
                                                    [
                                                        html.H3('Big Green, The (1995)', style={'fontSize': '20px', 'textAlign': 'center'}),
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/54.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating4', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,
                                                )
                                                
                                            ],
                                            className="object",
                                        ),
                                    
                                       dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H3('Two if by Sea (1996)', style={'fontSize': '20px', 'textAlign': 'center'}),  # Adjust font size and alignment
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/18.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating5', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,  
                                                ),
                                                
                                                dbc.Col(
                                                    [
                                                        html.H3('Big Bully (1996)', style={'fontSize': '20px', 'textAlign': 'center'}),
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/75.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating6', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,
                                                ),
                                                    dbc.Col(
                                                    [
                                                        html.H3('Nico Icon (1995)', style={'fontSize': '20px', 'textAlign': 'center'}), 
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/77.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating7', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3, 
                                                ),
                                                
                                                dbc.Col(
                                                    [
                                                        html.H3('Shopping (1994)', style={'fontSize': '20px', 'textAlign': 'center'}),
                                                        html.Img(src='https://liangfgithub.github.io/MovieImages/98.jpg', style={'width': '100%', 'height': '200px'}),
                                                        dcc.Input(id='rating8', type='number', min=1, max=5, step=1, value=1),
                                                    ],
                                                    md=3,
                                                )
                                                
                                            ],
                                            className="object",
                                        ),                                        dbc.Button(
                                            'Submit Ratings',
                                            id='Ratings Submission',
                                            n_clicks=0,
                                            color='primary',
                                            className="object",
                                        ),
                                    ]
                                ),
                                html.Div(id='Movie Recommendations2'),
                            ]
                        ),
                        html.H3("Here are some movie recommendations"),
                        html.Div(id='Movie Recommendations3'),
                    ],
                ),
            ]
        ),
    ],
    className="p-5",
)

@app.callback(
    Output('Movie Recommendations', 'children'),
    [Input('Display Movies', 'n_clicks')],
    [State('Genre Selection', 'value')]
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
                style={'margin-bottom': '20px'} 
            )
            for title, image_url in zip(selected_data['Title'], selected_data['image_url'])
        ]
        return movie_list
    return ""


@app.callback(
    Output('Movie Recommendations2', 'children'),
    [Input('Ratings Submission', 'n_clicks')],
    [State(f'rating{i}', 'value') for i in range(1, 9)]
)
def update_ratings_list(n_clicks, *ratings):
    global user_ratings
    if n_clicks > 0:
        user_ratings = list(ratings)
        return html.P(f'Ratings submitted: {user_ratings}')
    return []

def process_ratings_data(movie_data, user_ratings):
    combined_data = []

    for title, rating in zip(to_rate, user_ratings):
        combined_data.append({'MovieID': title, 'Rating': rating})

    return combined_data


@app.callback(
    Output('Movie Recommendations3', 'children'),
    [Input('Ratings Submission', 'n_clicks')],
    [State(f'rating{i}', 'value') for i in range(1, 9)]
)
def generate_movie_recommendations(n_clicks, *ratings):
    global user_ratings
    if n_clicks > 0:
        user_ratings = list(ratings)
        to_rate = ['m8','m37','m38','m54','m64','m75','m77','m98']
        combined = pd.DataFrame({'newuser':ratings}, index=to_rate)

        system2_df = get_system_data()
        newuser = system2_df.merge(combined, how='left', left_index=True, right_index=True)['newuser']
        predictions = myIBCF(newuser)
        
        movies2 = get_movies().loc[:,['MovieID','Title','image_url']]
        
        recommendations = movies2.merge(predictions, how = 'inner', left_on='MovieID',right_on ='movie_id')
        
        movie_recommendations = [
            html.Div(
                [
                    html.H3(title, style={'fontSize': '20px'}),
                    html.Img(src=image_url, style={'width': '200px', 'height': '200px'}),
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




