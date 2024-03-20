import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('tuned_random_forest_model.joblib')

# Load the scaler used
scaler = joblib.load('minmax_scaler.joblib')

# Create a Streamlit interface to take user input and make predictions
st.title('Loan Prediction Model Test and Deployment')

# Add input elements for the encoded features using selectbox and number_input
# feature_id = st.number_input('Number of Passengers')
feature_income = st.number_input('Purchase Lead Amount')
feature_age = st.number_input('Length of Stay (Hour)')
feature_experience = st.number_input('Flight Hour')
feature_marital_status = st.number_input('Route')
feature_house_ownership = st.number_input('Booking Origin')
feature_car_ownership = st.number_input('Wants Extra Baggage')
feature_profession = st.number_input('Wants Preferred Seats')
feature_city = st.number_input('Wants In-Flight Meals')
feature_state = st.number_input('Flight Duration')

# Generation
# Function to determine the generation based on age
def assign_generation(age):
    if age <= 27:
        return 'Generation Z'
    elif age <= 43:
        return 'Generation Millenials'
    elif age < 59:
        return 'Generation X'
    elif age < 69:
        return 'Boomers II'
    elif age <= 78:
        return 'Boomers I'
    else:
        return 'Other'

df_encoding['Generation'] = df_encoding['Age'].apply(assign_generation)

# Label Encoding for Generation
df_encoding['Generation'].replace({'Generation Z': 0,
                                   'Generation Millenials': 1,
                                   'Generation X' : 2,
                                   'Boomers II' : 3,
                                   'Boomers I' : 4,
                                   'Other' : 5},inplace=True)

df_encoding['Generation'] = df_encoding['Generation'].astype('int64')

# State
# Function for grouping state
def state_group(state) :
    if state in ['uttar pradesh', 'haryana', 'jammu and kashmir', 'punjab', 'uttarakhand', 'chandigarh', 'delhi', 'himachal pradesh'] :
        return 'north_zone'
    elif state in ['bihar', 'jharkhand', 'odisha', 'west bengal', 'assam', 'sikkim', 'tripura', 'mizoram', 'manipur'] :
        return 'east_zone'
    elif state in ['andhra pradesh', 'tamil nadu', 'karnataka', 'telangana', 'kerala', 'puducherry'] :
        return 'south_zone'
    else :
        return 'west_zone'

df_encoding['State_Group'] = df_encoding['State'].apply(state_group)

#One-hot Encoding for State grouping
onehots = pd.get_dummies(df_encoding['State_Group'], prefix='State')
onehots = onehots.astype(int)
df_encoding = pd.concat([df_encoding, onehots], axis=1)

#Drop the original State and State_Group after one-hot encoding
df_encoding.drop(columns=['State', 'State_Group'], inplace=True)

# City
# Function for grouping city
def city_group(city):
    if city in ['new delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore']:
        return 'metro'
    elif city in ['ahmedabad', 'hyderabad', 'pune', 'surat', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'visakhapatnam', 'indore', 'thane',
                  'bhopal', 'pimpri-chinchwad', 'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut', 'rajkot',
                  'varanasi', 'srinagar', 'amritsar', 'allahabad', 'jabalpur', 'gwalior', 'vijayawada', 'jodhpur', 'raipur', 'kota', 'guwahati', 'chandigarh city']:
        return 'urban'
    elif city in ['navi mumbai', 'kalyan-dombivli', 'vasai-virar', 'mira-bhayandar', 'thiruvananthapuram', 'bhiwandi', 'noida', 'bhopal', 'howrah', 'saharanpur',
                  'berhampur', 'suryapet', 'muzaffarpur', 'nadiad', 'siliguri', 'bhavnagar', 'kurnool', 'tenali', 'satna', 'nandyal', 'etawah', 'morena', 'ballia',
                  'machilipatnam', 'mau', 'machilipatnam', 'bhagalpur', 'siwan', 'meerut', 'dibrugarh', 'gaya', 'darbhanga', 'hajipur', 'mirzapur', 'akola', 'satna',
                  'motihari', 'jalna', 'ramgarh', 'ozhukarai', 'saharsa', 'munger', 'farrukhabad', 'nangloi jat', 'thoothukudi', 'nagercoil', 'rourkela', 'jhansi', 'sultan pur majra']:
        return 'suburban'
    else:
        return 'rural'

df_encoding['City_Group'] = df_encoding['City'].apply(city_group)

#One-hot Encoding for City grouping
onehots = pd.get_dummies(df_encoding['City_Group'], prefix='City')
onehots = onehots.astype(int)
df_encoding = pd.concat([df_encoding, onehots], axis=1)

#Drop the original City and City_Group after one-hot encoding
df_encoding.drop(columns=['City', 'City_Group'], inplace=True)


# When the user clicks the 'Predict' button, make predictions using the loaded model
if st.button('Predict'):
    # Organize the user inputs into a list or array
    user_inputs = [feature_income,
                   feature_age,
                   feature_experience,
                   feature_marital_status,
                   feature_house_ownership,
                   feature_car_ownership,
                   feature_profession,
                   feature_city,
                   feature_state
                   ]

    # Convert the user_inputs list to a numpy array
    user_inputs_array = np.array(user_inputs)

    # Reshape the array to make it 2D
    reshaped_inputs = user_inputs_array.reshape(1, -1)


    # Make predictions
    prediction = model.predict(reshaped_inputs)

    # Display the prediction
    st.success(f'This costumer will be {prediction[0]} (0: Not potential to complete booking, 1: Potential to complete booking)')