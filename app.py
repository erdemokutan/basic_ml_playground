#Core pkgs
import streamlit as st


#EDA pkgs
import pandas as pd
import numpy as np

#Data Viz pkgs
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
#ML pkgs
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC




def main():
    """My Simple Machine Learning Playground"""
    

    st.title("My Simple Machine Learning Playground")
    st.text("Using Streamlit==1.8.0+")
   
   
    st.subheader("This app might have problems in complex datasets.Please try to use simple datasets (for example: Iris)")
    

    activities = ["EDA","Plot","Model Building","About"]

    choice=st.sidebar.selectbox("Select Activity",activities)

    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis Section")

        data=st.file_uploader("Upload Your Dataset",type=["csv","txt"])
        st.text("tick the checkbox sequentially")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Show Shape"):
                st.write(df.shape)
            
            if st.checkbox("Show Columns"):
                all_columns=df.columns.to_list()
                st.write(all_columns)

            if st.checkbox("Select Columns to Show"):
                selected_columns=st.multiselect("Select Columns",all_columns)
                new_df=df[selected_columns]
                st.dataframe(new_df)
                
            
            if st.checkbox("Show Summary"):
                st.write(df.describe())
            
            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:,-1].value_counts())




    elif choice == 'Plot':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("Data Visualization Section")

        data=st.file_uploader("Upload Your Dataset",type=["csv","txt"])
        st.text("tick the checkbox sequentially")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head())

        if st.checkbox("Correlation with Seaborn"):
            st.write(sns.heatmap(df.corr(),annot=True))
            
            st.pyplot()


        if st.checkbox("Pie Chart"):
            all_columns=df.columns.to_list()
            column_to_plot=st.selectbox("Select 1 Column",all_columns)
            pie_plot=df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(pie_plot)
            st.pyplot()


        all_columns_names=df.columns.tolist()
        type_of_plot=st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
        st.text("Some Type of Plots might not work on all the data")
        selected_columns_names=st.multiselect("Select Columns To Plot",all_columns_names)

        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

            #Plot By Streamlit
            if type_of_plot=="area":
                cust_data=df[selected_columns_names]
                st.area_chart(cust_data)

            elif type_of_plot=="bar":
                cust_data=df[selected_columns_names]
                st.bar_chart(cust_data)

            elif type_of_plot=="line":
                cust_data=df[selected_columns_names]
                st.line_chart(cust_data)

            elif type_of_plot=="hist":
                cust_data=df[selected_columns_names]
                st.hist_chart(cust_data)

            elif type_of_plot=="box":
                cust_data=df[selected_columns_names]
                st.box_chart(cust_data)

            elif type_of_plot=="kde":
                cust_data=df[selected_columns_names]
                st.kde_chart(cust_data)

            else:
                st.write("Please Select a Valid Plot Type")


        


    elif choice == 'Model Building':
        st.subheader("Model Building Section")

        data=st.file_uploader("Upload Your Dataset",type=["csv","txt"])
        st.text("tick the checkbox sequentially")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head())

        #Model Building
        X=df.iloc[:,0:-1].values 
        Y=df.iloc[:,-1].values
        seed=7

        #Model
        models=[]
        models.append(("Logistic Regression",LogisticRegression()))
        models.append(("Linear Discriminant Analysis",LinearDiscriminantAnalysis()))
        models.append(("K-Nearest Neighbors Classifier",KNeighborsClassifier()))
        models.append(("Decision Tree Classifier",DecisionTreeClassifier()))
        models.append(("GaussianNB",GaussianNB()))
        models.append(("Support Vector Machine",SVC()))
        #evaluate each model in turn

        #List
        model_names=[]
        model_mean=[]
        model_std=[]
        all_models=[]
        scoring="accuracy"

        for name,model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=None)
            cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring) #cv_result=cross validation result
            model_names.append(name)
            model_mean.append(cv_results.mean())
            model_std.append(cv_results.std())

            accuracy_results={'model_name':name,'model_accuracy':cv_results.mean(),'standard_deviation':cv_results.std()}

            all_models.append(accuracy_results)

        if st.checkbox("Metrics as Table"):
            st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Model Name","Mean","Standard Deviation"]))

        if st.checkbox("Metrics as JSON"):
            st.json(all_models)

        
        

             
    
    elif choice == 'About':
        
        st.subheader("Created by Erdem Okutan")
        st.text("My Github:https://github.com/erdemokutan")
        st.text("My Email:erdem.okutan@hotmail.com")
        st.text("My LinkedIn:https://www.linkedin.com/in/erdemokutan/")
        st.text("My Instagram:https://www.instagram.com/erdemokutan/")
        st.text("My Twitter:https://twitter.com/zihinzugurdu")
        st.image("./silencewench.gif",width=100,caption="GIF",use_column_width=True)
        
        


if __name__ == '__main__':
    main()   
