import streamlit as st
import pandas    as pd
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler


DATASET = './dataset.csv'


def get_dataframe(file):
    # Create Pandas DataFrame from Dataset csv file
    return pd.read_csv(file)


def split_dataset(df):
    # Split the data into into independent 'X' and dependent 'Y' variables
    X = df.iloc[:, 0:9].values 
    Y = df.iloc[:,9].values

    # Split the dataset into 80% Training set and 20% Testing set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, Y_train, Y_test


@st.cache(allow_output_mutation=True)
def train_model(X_train, Y_train):
    # Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    print('Model Training Score (KNN):', knn.score(X_train, Y_train))
    return knn


if __name__ == '__main__':
    st.set_page_config(page_title='Breast Cancer predictor', layout='wide')
    st.title('Breast Cancer Prediction Web App')
    st.write("This app predicts whether a person has breast cancer or not based on clnical data.")
    st.write("This app uses the [Breast Cancer Coimbra Data set](https://www.kaggle.com/yasserhessein/breast-cancer-coimbra-classification-with-eda-ml/data)")
    st.write("❗**Disclaimer**: This is a simple web app that is not intended to be used for medical purposes.")
    
    df = get_dataframe(DATASET)
    X_train, X_test, Y_train, Y_test = split_dataset(df)
    model = train_model(X_train, Y_train)
    prediction = None
    
    if st.checkbox('Show Dataset'):
        st.dataframe(df)
        st.caption("Age (years) | BMI (kg/m2) | Glucose (mg/dL) | Insulin (µU/mL) | HOMA | Leptin (ng/mL) | Adiponectin (µg/mL) | Resistin (ng/mL) | MCP-1(pg/dL)")

    with st.sidebar:
        with st.form('user-input'):
            age = st.number_input('Age (years)', min_value=0)
            bmi = st.number_input('BMI (kg/m2)', min_value=0.0000, format="%.4f", step=0.0001)
            glucose = st.number_input('Glucose (mg/dL)', min_value=0)
            insulin = st.number_input('Insulin (µU/mL)', min_value=0.0000, format="%.4f", step=0.0001)
            homa = st.number_input('HOMA', format="%.4f", min_value=0.0000, step=0.0001)
            leptin = st.number_input('Leptin (ng/mL)', min_value=0.0000, format="%.4f", step=0.0001)
            adiponectin = st.number_input('Adiponectin (µg/mL)', min_value=0.0000, format="%.4f", step=0.0001)
            resistin = st.number_input('Resistin (ng/mL)', min_value=0.0000, format="%.4f", step=0.0001)
            mcp1 = st.number_input('MCP-1(pg/dL)', min_value=0.0000, format="%.4f", step=0.0001)
            if st.form_submit_button('Submit'):
                prediction = model.predict([[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp1]])[0]
    st.header('Result')
    if prediction:
        if prediction == 1:
            st.success('No Breast Cancer 🙌')
        else:
            st.warning('❗Breast Cancer Suspected, Please consult a physician')
    else:
        st.info('Please submit the data from sidebar')