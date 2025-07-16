import streamlit as st 
import pandas as pd 
import numpy as np
from sklearn.svm import SVC     
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score
)

# --- Page style and header ---
st.set_page_config(page_title="Visa Classifier App", layout="centered")
st.markdown("""
    <style>
    .title {
        font-size:32px;
        font-weight:bold;
        color:#4CAF50;
    }
    .footer {
        position: fixed;
        bottom: 10px;
        left: 0;
        right: 0;
        text-align: center;
        font-size: 14px;
        color: gray;
    }
    .main {
        background-color: #F7F9FC;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🔍 Visa Approval Classifier  - Make Your Choices on the Left Panel </div>', unsafe_allow_html=True)


def main():
    st.sidebar.title("Machine Learning Web App - Synthetic Data Used To Train Models")
    st.sidebar.markdown("Are you ready to explore the world of Machine Learning? This web app allows you to understand how your features affect visa approval outcomes. Let's get started!🇺🇸")

    with st.expander("📘 Click here to learn what these terms mean"):
        st.markdown("""
        **🔍 Key Machine Learning Terms Explained**

        - **Support Vector Machine (SVM):** A supervised learning model that finds the optimal boundary (hyperplane) to separate different classes.
        - **Random Forest:** A powerful ensemble method that builds multiple decision trees and merges their results to improve accuracy and avoid overfitting.
        - **Logistic Regression:** A statistical model used to predict binary outcomes (like visa approval or denial) based on input features.

        **📊 Model Evaluation Metrics**

        - **Accuracy:** The proportion of correct predictions made by the model.
        - **Precision:** Out of all predicted positive cases, how many were actually positive (Approved).
        - **Recall:** Out of all actual positive cases, how many were correctly predicted.
        - **Confusion Matrix:** A table showing correct and incorrect classifications across categories.
        - **ROC Curve:** A graph showing model performance across different threshold settings.
        - **Precision-Recall Curve:** Useful when dealing with imbalanced datasets; shows the trade-off between precision and recall.
        """)

    @st.cache_data(persist=True)
    def load_data():
        # data = pd.read_csv("/Users/gee/Library/Mobile Documents/com~apple~CloudDocs/Personal Projects/Book-of-Projects/Web Analytics/Web/Streamlit/Updated_Visa_Dataset.csv")
        # data = pd.read_csv("https://raw.githubusercontent.com/Gee1225/web_app/main/Updated_Visa_Dataset.csv")
        # label = LabelEncoder()
        # for col in data.columns:
        #     data[col] = label.fit_transform(data[col])
        # return data

        url = "https://raw.githubusercontent.com/Gee1225/web_app/main/Updated_Visa_Dataset.csv"
        try:
            data = pd.read_csv(url)
        except Exception as e:
            st.error(f"Failed to load {url}:\n{e}")
            st.stop()
        # …rest of your label‑encoding…
        return data


    @st.cache_data(persist=True)
    def split(df):
        y = df.approval_status
        x = df.drop(columns="approval_status")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test

    def plot_metricx(metrics, model, x_test, y_test, class_names):
        if "Confusion Matrix" in metrics:
            st.subheader("Confusion Matrix")
            disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names)
            st.pyplot(disp.figure_)

        if "ROC Curve" in metrics:
            st.subheader("ROC Curve")
            disp = RocCurveDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(disp.figure_)

        if "Precision_Recall_Curve" in metrics:
            st.subheader("Precision Recall Curve")
            disp = PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(disp.figure_)

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ["Denied", "Approved"]

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine", "Random Forest", "Logistic Regression"))

    if classifier == "Support Vector Machine":
        st.sidebar.subheader("Model Hyperparameters")

        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
        with st.sidebar.expander("ℹ️ What is C?"):
            st.markdown("""
            Controls the trade-off between misclassification and simplicity of the decision boundary.
            Smaller values specify stronger regularization.
            """)

        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        with st.sidebar.expander("ℹ️ What is a kernel?"):
            st.markdown("""
            Kernel functions map input data into higher-dimensional space to make separation easier.
            - **RBF:** Good for nonlinear patterns.
            - **Linear:** Best for linearly separable data.
            """)

        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")
        with st.sidebar.expander("ℹ️ What is Gamma?"):
            st.markdown("""
            Defines how far the influence of a single training example reaches.
            - **Low gamma:** smoother boundary.
            - **High gamma:** more complex boundary.
            """)

        Acc_metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision_Recall_Curve"))

        if st.sidebar.button("Classify", key='clasify'):
            st.subheader("📊 Support Vector Machine Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy_val = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            st.success(f"✅ Accuracy: {round(accuracy_val, 2)}")
            st.info(f"🎯 Precision: {round(precision_score(y_test, y_pred), 2)}")
            st.warning(f"📡 Recall: {round(recall_score(y_test, y_pred), 2)}")

            plot_metricx(Acc_metrics, model, x_test, y_test, class_names)

    # if st.sidebar.checkbox("Show raw data", False):
    #     st.subheader("Visa Classification Data")
    #     st.write(df)

    # if st.sidebar.checkbox("Show dataset summary"):
    #     st.subheader("📊 Dataset Summary")
    #     st.write(df.describe())
    #     st.write("Class Distribution:")
    #     st.bar_chart(df['approval_status'].value_counts())

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")

        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
        with st.sidebar.expander("ℹ️ What is C?"):
            st.markdown("""
            Controls the trade-off between misclassification and simplicity of the decision boundary.
            Smaller values specify stronger regularization.
            """)

        max_iter = st.sidebar.slider("Maximum number of iterations",100,500, key="max_iter")
        with st.sidebar.expander("ℹ️ What is a maximum iterations?"):
            st.markdown("""
            This is the maximum number of iterations for the solver to converge.
            A higher value allows the model to learn more complex patterns.
            """)

        Acc_metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision_Recall_Curve"))

        if st.sidebar.button("Classify", key='clasify'):
            st.subheader("📊 Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy_val = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            st.success(f"✅ Accuracy: {round(accuracy_val, 2)}")
            st.info(f"🎯 Precision: {round(precision_score(y_test, y_pred), 2)}")
            st.warning(f"📡 Recall: {round(recall_score(y_test, y_pred), 2)}")

            plot_metricx(Acc_metrics, model, x_test, y_test, class_names)

    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")

        n_estimator= st.sidebar.number_input("N_estimator", 100, 500, step=10, key="n_estimator")
        with st.sidebar.expander("ℹ️ What is n_estimator?"):
            st.markdown("""
            This controls the number of trees in the forest. More trees can improve performance but also increase computation time.
            """)

        max_depth = st.sidebar.number_input("Maximum depth of trees",1,20,step=1, key="max_depth")
        with st.sidebar.expander("ℹ️ What is a maximum depth?"):
            st.markdown("""
            This is the maximum number of iterations for the solver to converge.
            A higher value allows the model to learn more complex patterns.
            """)

        bootstrap = st.sidebar.radio("Number of bootstrap samples when building trees:",(True,False),key="bootstrap")
        with st.sidebar.expander("ℹ️ What is a bootstrap?"):
            st.markdown("""
            This is the maximum number of iterations for the solver to converge.
            A higher value allows the model to learn more complex patterns.
            """)

        Acc_metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision_Recall_Curve"))

        if st.sidebar.button("Classify", key='clasify'):
            st.subheader("📊 Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth,bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy_val = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            st.success(f"✅ Accuracy: {round(accuracy_val, 2)}")
            st.info(f"🎯 Precision: {round(precision_score(y_test, y_pred), 2)}")
            st.warning(f"📡 Recall: {round(recall_score(y_test, y_pred), 2)}")

            plot_metricx(Acc_metrics, model, x_test, y_test, class_names)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Visa Classification Data")
        st.write(df)

    if st.sidebar.checkbox("Show dataset summary"):
        st.subheader("📊 Dataset Summary")
        st.write(df.describe())
        st.write("Class Distribution:")
        st.bar_chart(df['approval_status'].value_counts())    

    st.markdown('<div class="footer">Made with ❤️ by Gee | Powered by Streamlit & scikit-learn</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
