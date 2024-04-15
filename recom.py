
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
        rec = filtered_data.iloc[car_indices.flatten()[1:6]]


        return rec[['manufacturer','model','condition', 'cylinders','year','price']]
    
    else:
        # Sort cars based on optimal value (e.g., manufacturing year, odometer)
        sorted_cars = filtered_data.sort_values(by=['years', 'odometers'], ascending=[False, True])

        # Return the top 5 cars with the best optimal value
        best_cars = sorted_cars.head(5)
        return best_cars


