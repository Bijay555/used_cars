# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# def recommend_func(data, min_price, max_price, **kwargs):
#     '''
#     data set: car_data
#     parameters: price_range, **kwargs (optional: year, odometer, manufacturer, paint_color, car_type, and additional parameters)
#     return: dataframe containing the top similar cars
#     '''
#     # Apply filters based on provided parameters
#     for key, value in kwargs.items():
#         if value:  # Check if value is provided
#             if key == 'year':
#                 data = data[data[key] == value + ' AD']
#             elif key == 'odometer':
#                 data = data[data[key] == value + ' miles']
#             elif key == 'cylinders':
#                 data = data[data[key] == value + ' cylinders']
#             else:
#                 data = data[data[key] == value]

#     # Filter data based on price_range
#     data = data[(data['price'] >= min_price) & (data['price'] <= max_price)]
    

#     if kwargs:
#         # Perform similarity calculations based on car details
#         tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df = 1, stop_words='english',token_pattern=r'\w{1,}')
#         tfidf_matrix = tfidf_vectorizer.fit_transform(data['car_details'])
#         similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

#         # Get indices of top similar cars
#         car_indices = similarity_matrix.argsort()[:, ::-1][:, :-7:-1]  # Exclude self-similarity and get top 6 similar indices

#         # Recommendation for top 6 similar cars
#         rec = data.iloc[car_indices.flatten()]
#         return rec
    
#     else:
#         # Sort cars based on optimal value (e.g., manufacturing year, odometer)
#         sorted_cars = data.sort_values(by=['years', 'odometers'], ascending=[False, True])

#         # Return the top 5 cars with the best optimal value
#         best_cars = sorted_cars.head(5)
#         return best_cars


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_func(data, min_price, max_price, **kwargs):
    '''
    data set: car_data
    parameters: price_range, **kwargs (optional: year, odometer, manufacturer, paint_color, car_type, and additional parameters)
    return: dataframe containing the top similar cars
    '''
    # Apply filters based on provided parameters
    filtered_data = data.copy()  # Make a copy of the original data to avoid modifying it
    for key, value in kwargs.items():
        if value:  # Check if value is provided
            if key == 'year':
                print(value)
                value = value + ' AD'
                print(value)
                filtered_data = filtered_data[filtered_data[key] == value]
            elif key == 'odometer':
                value = value +  ' miles'
                filtered_data = filtered_data[filtered_data[key] == value]
            elif key == 'cylinders':
                value = value + ' cylinders'
                filtered_data = filtered_data[filtered_data[key] == value]
            else:
                filtered_data = filtered_data[filtered_data[key] == value]

    # Filter data based on price_range
    filtered_data = filtered_data[(filtered_data['price'] >= min_price) & (filtered_data['price'] <= max_price)]
    print(filtered_data)
    if kwargs:
        if len(filtered_data) == 0:
            return None  # Return None if no data matches the filters
        
        # Perform similarity calculations based on car details
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_data['car_details'])
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Get indices of top similar cars
        car_indices = similarity_matrix.argsort()[:, ::-1]  # Exclude self-similarity and get top 5 similar indices

        # Recommendation for top 6 similar cars
        rec = filtered_data.iloc[car_indices.flatten()[1:7]]

        return rec
    
    else:
        # Sort cars based on optimal value (e.g., manufacturing year, odometer)
        sorted_cars = filtered_data.sort_values(by=['year', 'odometer'], ascending=[False, True])

        # Return the top 5 cars with the best optimal value
        best_cars = sorted_cars.head(5)
        return best_cars


