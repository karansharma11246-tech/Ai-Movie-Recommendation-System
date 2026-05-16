import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = {
    'title': ['Avatar', 'Titanic', 'Interstellar', 'Inception', 'Gravity'],
    'genre': ['Sci-Fi', 'Romance', 'Sci-Fi', 'Sci-Fi', 'Sci-Fi']
}

df = pd.DataFrame(movies)

cv = CountVectorizer()
matrix = cv.fit_transform(df['genre'])

similarity = cosine_similarity(matrix)

def recommend(movie):
    idx = df[df['title'].str.lower() == movie.lower()].index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_movies = sorted(scores, key=lambda x: x[1], reverse=True)

    print("Recommended Movies:")
    for i in sorted_movies[1:]:
        print(df.iloc[i[0]].title)

movie = input("Enter movie name: ")
recommend(movie)