from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_func(data, min_price, max_price, **kwargs):
    '''
    data set: car_data
    parameters: price_range, **kwargs (optional: manufacturer, paint_color, car_type, and additional parameters)
    return: dataframe containing the top similar cars
    '''
    # Apply filters based on provided parameters
    for key, value in kwargs.items():
        if value:  # Check if value is provided
            data = data[data[key] == value]

    # Filter data based on price_range
    data = data[(data['price'] >= min_price) & (data['price'] <= max_price)]
    
    if kwargs:
        # Perform similarity calculations based on car details
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['car_details'])
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Get indices of top similar cars
        car_indices = similarity_matrix.argsort()[:, ::-1][:, 1:7]  # Exclude self-similarity and get top 6 similar indices

        # Recommendation for top 6 similar cars
        rec = data.iloc[car_indices.flatten()]
        return rec
    
    else:
                # Sort cars based on optimal value (e.g., manufacturing year, odometer)
        sorted_cars = data.sort_values(by=['year', 'odometer'], ascending=[False, True])

        # Return the top 5 cars with the best optimal value
        best_cars = sorted_cars.head(5)
        return best_cars

