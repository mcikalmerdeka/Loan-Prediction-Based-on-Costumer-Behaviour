import streamlit as st
import streamlit.components.v1 as stc

from ml_app import run_ml_app

html_temp = """
            <div style="background-color:#2E8B57;padding:20px">
		    <h1 style="color:white;text-align:center;font-family: Arial, sans-serif;">Loan Prediction App</h1>
		    <h2 style="color:white;text-align:center;font-family: Arial, sans-serif;">Dackers Lending Company</h2>
	    </div>
            """

desc_temp = """
            ### Loan Prediction App
            This app will be used for business purposes to predict whether a customer is highly potential of default or not.
            #### Data Source
            - https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior
            #### App Content
            - Home Section
            - Machine Learning Prediction Section
            """

def main():

    stc.html(html_temp)

    menu = ['Home', 'Machine Learning']
    with st.sidebar:
        stc.html("""
                    <style>
                        .circle-image {
                            width: 130px;
                            height: 130px;
                            border-radius: 50%;
                            overflow: hidden;
                            box-shadow: 0 0 10px rgba(1, 1, 1, 1);
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            background-color: white;
                        }
                        
                        .circle-image img {
                            width: 100%;
                            height: 100%;
                            object-fit: contain;
                        }
                    </style>
                    <div class="circle-image">
                        <img src="https://www.nicepng.com/png/detail/270-2702624_personal-loans-finanas-icon.png"> 
                    </div>
                    """
                )
        st.subheader('Loan Prediction App')
        st.write("---")
        choice = st.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader("Welcome to Homepage")
        st.write("---")
        st.markdown(desc_temp)
    elif choice == "Machine Learning":
        # st.subheader("Welcome to Machine learning")
        run_ml_app()


if __name__ == '__main__':
    main()