import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df=pd.read_csv("selected_train.csv")
explanations=pd.read_csv("explanations.csv")
# Extracting X and y from the dataset
X = df.drop(columns="READMITTED_7_DAYS")
y = df["READMITTED_7_DAYS"]
# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1,stratify=y)
# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Define the logistic regression equation coefficients
coefficients = [-1.9551, 0.1717, -0.1457, 0.3259, -0.5330, 0.1611, 0.7481, 0.2675, -0.1730,-0.3356, 0.1791, -0.3301, 0.1403, 0.3896, -0.2794, -0.4268, -0.2632, -0.4738,0.2858, 0.2345, -0.2607, 0.3920, -0.2864, 0.7884, 0.7250, -2.3699]
# Define the minimum and maximum values for each variable
average_values=[float(value) for value in X.mean().values]
min_values = [12.00, 0.17, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,0.00, 0.00, 1.00, 1.00, 0.30, 30.00, 30.70, 1.00, 5.78, 54.00, 0, 0, 0, 0]
max_values = [113.000, 500, 1.000, 1.000, 4.000, 4.000, 4.000, 4.000, 24.000,4.000, 4.000, 43.498, 150.000, 151.500, 166.000, 6.500, 40.000,40.800, 192.750, 59.733, 300.000, 135.000, 1.000, 1.000, 1.000]
feature_names = ['AGE', 'LOS', 'TYPE', 'MV_24_HOUR', 'INITIAL_SOFA_CARDIO', 'DISCHARGE_SOFA_NERV', 'DISCHARGE_SOFA_RESPIRATORY', 'MAX_SOFA_CARDIO', 'MAX_SOFA', 'AVER_SOFA_LIVER', 'AVER_SOFA_NERV', 'VAR_SOFA', 'DISCHARGEAPACHE', 'MEANAPACHE', 'MAXAPACHE', 'ALBUMIN', 'MIN_TEMP', 'MAX_TEMP', 'BLOOD_UREA_NITROGEN', 'HEMATOCRIT', 'MAX_HEART_RATE', 'MIN_HEART_RATE', 'DEST_LEVEL_OF_CARE_Regular', 'DEST_LEVEL_OF_CARE_SDU', 'DEST_LEVEL_OF_CARE_out']
# Function to calculate the probability using the logistic function
def calculate_probability(inputs):
    logit = np.dot(inputs, coefficients[1:]) + coefficients[0]
    probability = 1 / (1 + np.exp(-logit))
    return probability
def display_probability_loading_bar(probability):
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(101):
        progress_bar.progress(i)
        status_text.text(f"Probability: {i:.3f}%")
        time.sleep(0.01)  # Just to simulate some progress

        # Break the loop when the desired probability is reached
        if i >= probability * 100:
            break
    status_text.text(f"Probability: {(probability[0] * 100).round(3)}%")
def plot_histogram(variable):
    grouped_data = df.groupby('READMITTED_7_DAYS')[variable].mean().reset_index()
    plt.figure(figsize=(8,6))  # Adjust the size as needed
    ax = sns.barplot(data=grouped_data, x='READMITTED_7_DAYS', y=variable)
    plt.xlabel(variable)
    plt.ylabel('Mean Readmission')
    plt.title(f'Mean Readmission for {variable}')
    # Add value annotations to the bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2, height/2), ha='center', va='center',
                        xytext=(0, 5), textcoords='offset points')

    st.pyplot(plt)
def plot_violin(variable):
    plt.figure(figsize=(8, 6))  # Adjust the size as needed
    sns.violinplot(data=df, x='READMITTED_7_DAYS', y=variable)
    plt.xlabel('Readmitted 7 Days')
    plt.ylabel(variable)
    plt.title(f'Violin Plot: {variable} vs Readmitted 7 Days')

    st.pyplot(plt)
############################
# PAGE CUSTOMIZATION
st.set_page_config(page_title="Readmission ML", page_icon="⚕️", layout="wide")
##########
page = st.sidebar.selectbox("Select a page", ["Main Page", "Probability", "Graphs", "Variables"])
if page == "Main Page":
    st.header("WELCOME TO READMISSION PREDICTION MODEL")
    st.markdown("#### In this page a brief explanation about each page will be given.")
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
    ### Probability Explanation
        st.markdown('<h2 style="color:red; font-size:25px'
                '">The Probability page</h2>',unsafe_allow_html=True)
        st.markdown("* Input their desired features and calculate the probability of ICU readmission based on the provided inputs.")
        st.markdown("* Divide the page into sections, with range sliders provided for adjusting the values of the corresponding features.")
        st.markdown("* Note that the last three features, denoted by the 'DEST' prefix, can only take binary values of 0 or 1, representing different levels of care for destination options.")
        st.markdown("* Obtain accurate predictions of ICU readmission probability by selecting the appropriate options for these features.")
    with col2:
        ### Graph Explanation
        st.markdown('<h2 style="color:red; font-size:25px">The Graphs page</h2>', unsafe_allow_html=True)
        st.markdown("* Select a variable from an input bar with a scrollbar.")
        st.markdown("* Plot a bar graph for the selected variable.")
        st.markdown("* Plot a violin graph for an independent variable.")
        st.markdown("* Obtain visual insights into the data distribution and relationships.")
    with col3:
        ### Variable Explanation
        st.markdown(
            '<h2 style="color:red; font-size:25px">The Variables page</h2>',
            unsafe_allow_html=True)
        st.markdown("* The coefficients and effects of each variable on the outcome.")
        st.markdown("* The nature of the effect, represented by color (green for an increase in the "
                    "probability of readmission and red for a decrease).")
        st.markdown("* A table displaying the coefficients and effects for each variable.")
        st.markdown("* Gain insights into the impact of different variables on the prediction model.")
elif page == "Probability":
    # Set up the Streamlit app layout
    st.title("ICU Readmission Prediction Model")
    col1, col2,col3,col4,col5 = st.columns(5,gap="small")
    with col1:
        st.header("Input Features_1")
        # Create range sliders for each feature
        num_features_1 = 6
        inputs = []
        for i in range(num_features_1):
            feature_name = feature_names[i]
            feature_range = (float(min_values[i]), float(max_values[i]))  # Convert to float
            feature_step = float(max_values[i] - min_values[i]) / 100  # Compute step based on range
            feature_value = st.slider(feature_name, *feature_range,step=feature_step,value=average_values[i])
            inputs.append(feature_value)
    with col2:
        st.header("Input Features_2")
        num_features_2 = 12
        for i in range(num_features_1,num_features_2):
            feature_name = feature_names[i]
            feature_range = (float(min_values[i]), float(max_values[i]))  # Convert to float
            feature_step = float(max_values[i] - min_values[i]) / 100  # Compute step based on range
            feature_value = st.slider(feature_name, *feature_range,step=feature_step,value=average_values[i])
            inputs.append(feature_value)
    with col3:
        st.header("Input Features_3")
        num_features_3 = 18
        for i in range(num_features_2,num_features_3):
            feature_name = feature_names[i]
            feature_range = (float(min_values[i]), float(max_values[i]))  # Convert to float
            feature_step = float(max_values[i] - min_values[i]) / 100  # Compute step based on range
            feature_value = st.slider(feature_name, *feature_range,step=feature_step,value=average_values[i])
            inputs.append(feature_value)
    with col4:
        st.header("Input Features_4")
        num_features_4 = 25
        select_options = ["Regular", "SDU", "out", "None"]
        for i in range(num_features_3, num_features_4):
            feature_name = feature_names[i]
            if feature_name in ["DEST_LEVEL_OF_CARE_Regular", "DEST_LEVEL_OF_CARE_SDU", "DEST_LEVEL_OF_CARE_out"]:
                st.write("Only one of these features can take the value of one")
                selected_option = st.selectbox(feature_name,options=[0,1])
                inputs.append(selected_option)
            else:
                feature_range = (float(min_values[i]), float(max_values[i]))  # Convert to float
                feature_step = float(max_values[i] - min_values[i]) / 100  # Compute step based on range
                feature_value = st.slider(feature_name, *feature_range, step=feature_step,value=average_values[i])
                inputs.append(feature_value)

    with col5:
        st.header("Probability of Being Readmitted")
        # Store input values in an array
        inputs = scaler.transform(np.array(inputs).reshape(1,-1))
        # Calculate the probability based on the inputs
        probability = calculate_probability(inputs)
        # Example usage
        display_probability_loading_bar(probability)
elif page == "Graphs":
    col1, col2= st.columns(2, gap="large")
    with col1:
        st.subheader("Bar Graph")
        # Select the variable from an input bar with scrollbar
        selected_variable = st.selectbox('Select a Variable', feature_names)

        # Plot the histogram for the selected variable
        if selected_variable in feature_names:
            plot_histogram(selected_variable)
        else:
            st.error("Invalid variable selection!")
    with col2:
        st.subheader("Violin Graph")
        selected_variable_2 = st.selectbox('Select an independent variable', feature_names)

        # Plot the histogram for the selected variable
        if selected_variable_2 in feature_names:
            plot_violin(selected_variable_2)
        else:
            st.error("Invalid variable selection!")
elif page == "Variables":
    st.markdown("# Variables Table")
    st.markdown("The color of the Coefficient represents the nature of its effect on the outcome ")
    st.markdown(
        """
        <style>
        .green-info {
            background-color: navy blue;
            color: green;
        }

        .red-info {
            background-color: navy blue;
            color: red;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="green-info">GREEN: INCREASE IN THE PROBABILITY OF READMISSION</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="red-info">RED: DECREASE IN THE PROBABILITY OF READMISSION</div>',
                unsafe_allow_html=True)

    effects = []
    for i in range(1, len(coefficients)):
        effect = np.exp(coefficients[i]) - 1
        effects.append("{:.4f}".format(effect))
    dataframe=pd.DataFrame(data={"Coefficients": coefficients[1:26],
                                 "Effect of a one-unit change in each variable on the outcome":Effects,
                                 "Description": explanations["Explanations"].values},
                           index=feature_names)
    styler = dataframe.style


    # Define a function to highlight positive values in green and negative values in red
    def highlight_positive(value):
        if value > 0:
            return "color: green"
        else:
            return "color: red"


    # Apply the highlighting function to the desired columns
    styled_dataframe = styler.applymap(highlight_positive, subset=["Coefficients"])

    # Display the styled DataFrame using st.table
    st.table(styled_dataframe)
