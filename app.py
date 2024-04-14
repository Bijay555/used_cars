import streamlit as st
import shap
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import pickle
from recom import recommend_func

import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore")

# # Set a background image URL
# background_image = 'https://raw.githubusercontent.com/Bijay555/used_cars/prod/images/car.jpg'
# html_code  = f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
# background-image: url('{background_image}');
# background-size: cover;
# background-position: top left;
# background-repeat:repeat;
# background-attachment: local;
# }}
# </style>
# """

# # Display the HTML code using st.markdown
# st.markdown(html_code, unsafe_allow_html=True)

# Set a background image URL
background_image = 'https://raw.githubusercontent.com/Bijay555/used_cars/prod/images/car.jpg'
html_code = f"""
<style>
/* Background styling */
[data-testid="stAppViewContainer"] > .main {{
    background-image: url('{background_image}');
    background-size: 90%; /* Ensure the background image covers the entire container */
    background-position: left; /* Center the background image */
    background-repeat: no-repeat; /* Prevent the background image from repeating */
    background-attachment: fixed; /* Fixed background position as the user scrolls */
}}

/* Page content styling */
.stAppView > div > div > div > div > div {{
    background-color: rgba(255, 255, 255, 0.9); /* Add a semi-transparent white overlay to improve readability */
    padding: 20px; /* Add padding to the content */
    border-radius: 10px; /* Add border radius for rounded corners */
    box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1); /* Add a subtle box shadow for depth */
}}

/* Title styling */
.stAppView > div > div > div > div > div > div > div > div > div > div > h1 {{
    color: #333; /* Set title color */
    text-align: center; /* Center align the title */
}}

/* Button styling */
.stButton>button {{
    background-color: #4CAF50; /* Green */
    border: none;
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 8px; /* Add border radius for rounded buttons */
}}
</style>
"""

# Display the HTML code using st.markdown
st.markdown(html_code, unsafe_allow_html=True)


# importing dataset
df = pd.read_csv('dataset/car_data.csv')

def home_page():
    st.title("Used Car Marketplace")
    st.write("### Welcome to the Used Car Marketplace!")

    # Button to go to Buyer page
    if st.button("Buyer"):
        st.session_state.page = "buyer"

    # Button to go to Seller page
    if st.button("Seller"):
        st.session_state.page = "seller"

def buyer_page():
    if st.button("Back to Home"):
        st.session_state.page = "home"

    st.write("# Buyer Page")
    st.write('---')
    st.write("### Best Cars to buy on :")
    estimated_range = st.text_input("Enter estimated buying range (min, max):", "")
    if estimated_range:
        min_price, max_price = map(int, estimated_range.split(','))

    # Define the list of columns
    columns = ['manufacturer', 'year', 'paint_color','condition', 'cylinders','state','drive']

    # Ask the user to select which columns they want to fill
    selected_columns = st.multiselect('Select Columns to Fill', columns)

    # Create a dictionary to store column values
    column_values = {}

    # For each selected column, create an input field
    for column in selected_columns:
        value = st.text_input(f'Enter value for {column}', key=column)

        # Store the value in the dictionary
        column_values[column] = value

    # Add a button to trigger processing
    if st.button("Process Data"):
        # Pass the column_values dictionary to the function
        recommended_cars = recommend_func(df, min_price, max_price, **column_values)
        if recommended_cars is not None and not recommended_cars.empty:
            st.write("Recommended cars based on your preferences:")
            st.write(recommended_cars)
        else:
            st.write("No cars found matching your criteria.")



    

def seller_page():
    st.write("# Seller Page")


    # Sidebar
    # Header of Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    car_manufacturer = st.sidebar.selectbox("Car Manufacturer:", ['select your car brand name','toyota', 'chrysler', 'ford', 'chevrolet', 'volkswagen', 'ram', 'gmc',
                                                         'cadillac', 'lexus', 'honda', 'bmw', 'pontiac', 'buick', 'dodge',
                                                         'nissan', 'subaru', 'jeep', 'hyundai', 'acura', 'mitsubishi', 'mazda',
                                                         'volvo', 'saturn', 'mercedes-benz', 'mini', 'rover', 'lincoln', 'audi',
                                                         'infiniti', 'jaguar', 'porsche', 'kia'], index=0)
    
    manufacturing_year = st.sidebar.number_input("Manufacturing Year:", min_value=1900, max_value=2022, value=1900)
    # Check if input year is provided
    if manufacturing_year == 0 or manufacturing_year < 1900 or manufacturing_year > 2022:
        st.sidebar.warning("Please enter a valid manufacturing year between 1900 and 2022.")
    else:
        years_from_2023 = 2023 - manufacturing_year

    condition = st.sidebar.selectbox("Condition:", ['how is your car condition?','excellent', 'good', 'fair', 'like new', 'salvage', 'new'], index=0)
    cylinders = st.sidebar.selectbox("Cylinders:", ['choose number of cylinders',3, 4, 5, 6, 8,10,12], index=0)
    type = st.sidebar.selectbox("Car Type:", ['select your car type','SUV', 'mini-van', 'truck', 'pickup', 'sedan', 'van', 'coupe', 'convertible',
                                          'hatchback', 'offroad', 'bus'], index=0)
    state = st.sidebar.selectbox("State:", ['select your state','ca', 'sc', 'pa', 'il', 'wi', 'az', 'or', 'mt', 'id', 'va', 'al', 'tx', 'ky', 'nc',
                                     'co', 'wy', 'fl', 'oh', 'la', 'ga', 'mn', 'de', 'in', 'mi', 'wa', 'nm', 'nv', 'tn',
                                     'nd', 'nj', 'ok', 'dc', 'ny', 'md', 'ms', 'mo', 'ia', 'ar', 'ks', 'me', 'ak', 'ri',
                                     'ne', 'ct', 'vt', 'nh', 'sd', 'ma', 'wv', 'ut', 'hi'], index=0)
    fuel = st.sidebar.selectbox("Fuel Type:", ['select your car fuel type','gas', 'diesel', 'hybrid', 'electric'], index=0)
    title_status = st.sidebar.selectbox("Title Status:", ['select your car status','clean', 'salvage', 'rebuilt', 'parts only', 'lien'], index=0)
    transmission = st.sidebar.selectbox("Transmission:", ['select transmission','automatic', 'manual'], index=0)
    drive = st.sidebar.selectbox("Drive:", ['select drive:','rwd', 'fwd', '4wd'], index=0)
    size = st.sidebar.selectbox("Size:", ['select car size:','mid-size', 'full-size', 'compact', 'sub-compact'], index=0)
    paint_color = st.sidebar.selectbox("Paint Color:", ['car color:','silver', 'blue', 'white', 'brown', 'black', 'grey', 'red', 'green', 'custom',
                                                 'yellow', 'purple', 'orange'], index=0)
    distance_travelled_in_miles = st.sidebar.number_input("Distance travelled (in miles):", min_value=0, value=0)
    # Check if input odometer is provided
    if distance_travelled_in_miles <= 0:
        st.sidebar.warning("Please enter a valid distance travelled in miles.")
    
    # Check if inputs are provided for all features
    if (car_manufacturer == 'select your car brand name' or
        condition == 'how is your car condition?' or
        cylinders == 'choose number of cylinders' or
        type == 'select your car type' or
        state == 'select your state' or
        fuel == 'select your car fuel type' or
        title_status == 'select your car status' or
        transmission == 'select transmission' or
        drive == 'select drive:' or
        size == 'select car size:' or
        paint_color == 'car color:'):
        st.sidebar.warning("Please select valid options for all features.")

  
    # Create a dummy DataFrame with seller input
    data = {
        "year": [years_from_2023],
        "manufacturer": [car_manufacturer],
        "condition": [condition],
        "cylinders": [cylinders],
        "fuel": [fuel],
        "odometer": [distance_travelled_in_miles],
        "title_status": [title_status],
        "transmission": [transmission],
        "drive": [drive],
        "size": [size],
        "type": [type],
        "paint_color": [paint_color],
        "state": [state]
        }
    
    
    #creating a user input dataframe
    user_input = user_input_features(data)
    st.write('### The user input :')
    st.write(user_input)

    # encoding categorical into numerical values
    encoded_input = encode(user_input)
    st.write('---')
    st.write('### The encoded user input :')

    st.write(encoded_input)
    st.write('---')

    st.write('### Estimated Price Prediction : ')

    if st.button("Predict"):
        # Load pre-trained model and calculate SHAP values
        #model = load_pickle(r'/home/bijay/used_car_web_app/used_cars/model/modelLgb.pkl')  # Define this function to load your pre-trained model
        lightgbm_model = pickle.load(open(r'model/modelLgb.pkl','rb'))

        prediction = lightgbm_model.predict(encoded_input)
        st.success('The estimated price is {} US Dollar'.format(prediction[0]))

        shap_values = shap_explanation(lightgbm_model, encoded_input)

        # Explain the model prediction using SHAP
        shap_values = shap_explanation(lightgbm_model, encoded_input)
    
        # Plot SHAP summary plot
        st.write("SHAP Summary Plot:")
    
        shap.summary_plot(shap_values, encoded_input)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(bbox_inches='tight')
        st.write('---')

        st.header('Feature Importance')

        plt.title('Feature importance based on SHAP values (Bar)')
        shap.summary_plot(shap_values, encoded_input, plot_type="bar")
        st.pyplot(bbox_inches='tight')
    
    if st.button("Back to Home"):
        st.session_state.page = "home"


def user_input_features(user_input):
    features = pd.DataFrame(user_input, index=[0])
    return features

@st.cache_data
def recommend_car(min_price, max_price,state, dataset):
    # Filter cars within the price range
    filtered_cars = dataset[(dataset['price'] >= min_price) & (dataset['price'] <= max_price) & (dataset['state'] == state) ]

    if filtered_cars.empty:
        print("No cars found within the given price range.")
        return None

    # Sort cars based on optimal value (e.g., manufacturing year, distance traveled, car condition)
    sorted_cars = filtered_cars.sort_values(by=['year', 'odometer'], ascending=False)

    # Return the car with the best optimal value
    best_car = sorted_cars.iloc[0:5]
    return best_car

@st.cache_data
def encode(data):
    '''function to encode data''' 
    # dropping numerical features
    categorical_columns = data.drop(['year','odometer', 'cylinders'], axis=1)
    df = categorical_columns.values
    impute_reshape = df.reshape(-1, 1)
    # Encode data
    impute_ordinal = OrdinalEncoder().fit_transform(impute_reshape)
    # Squeeze the resulting array to remove single-dimensional entries
    encoded_data = np.squeeze(impute_ordinal)
    # Replace original categorical columns with encoded values
    data_encoded = data.copy()  # Create a copy to avoid modifying the original DataFrame
    data_encoded[categorical_columns.columns] = encoded_data
    return data_encoded

# Load pickle object function
@st.cache_resource
def load_pickle(file):
    with open(file, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object


def shap_explanation(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values



def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "buyer":
        buyer_page()
    elif st.session_state.page == "seller":
        seller_page()

if __name__ == "__main__":
    main()
