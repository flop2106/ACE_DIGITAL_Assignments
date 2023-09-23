# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 21:13:43 2023

@author: nurar
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, f_oneway
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from collections import Counter
import re
       
class Q1:
    def __init__(self):
        
        self.data = pd.read_csv('ingredient.csv')
    
    def a(self):
        
        
        # Summary of the dataframe
        self.summary = self.data.describe()
        
        # Check for Normality (Parametric/Non-Parametric)
        self.normality_test_results = {}
        for column in self.data.columns:
            _, p_value = shapiro(self.data[column]) #Shapiro test with null hypothesis that data is normally distributed (parametric), alternative hypothersis is data is not normally distributed (non-parametric)
            self.normality_test_results[column] = p_value #all p-value are < 0.05 thus null hypothesis is rejected.
        
        #Correlation Analysis
        self.correlation_matrix = self.data.corr(method = 'spearman') #spearman as correlation are non-parametric
        
        #Anova Test
        self.anova_results = f_oneway(*[self.data[col] for col in self.data.columns])
        
        
    def b(self):

        # Distribution Study
        for column in self.data.columns:
            sns.histplot(self.data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.show()
        
        # Boxplot for Outlier Detection
        sns.boxplot(data=self.data)
        plt.title('Boxplot of all columns')
        plt.show()
        
        # Correlation Heatmap
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()
    
    def c(self):
        
        # Preprocessing: Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.data)
        
        # Dimensionality Reduction: Reduce to 3 principal components for visualization
        pca = PCA(n_components=3)
        data_pca = pca.fit_transform(data_scaled)
        
        # Determine the number of clusters using the Elbow Method
        inertia = []
        for i in range(1, 11):  # for example, check from 1 to 10 clusters
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(data_pca)
            inertia.append(kmeans.inertia_)
        
        plt.plot(range(1, 11), inertia)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()
        
        # Choose the optimal number of clusters (k) where the elbow occurs, for example, k=3
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(data_pca)
        self.data['cluster'] = clusters
        
        
        # Visualize the clusters in 3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=clusters, cmap='viridis')
        ax.set_title('3D PCA - Clusters')
        
        # Create a custom legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}', 
                                  markersize=10, markerfacecolor=plt.cm.viridis(i / optimal_k)) for i in range(optimal_k)]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.show()

class Q2:
    def __init__(self):

        self.data = pd.read_csv('palm_ffb.csv')
        
    def answer(self):
 
            # Running descriptive statistics
            print(self.data.describe())
            
            # Checking for missing values
            print(self.data.isnull().sum())
            
            x_vars = ['SoilMoisture', 'Average_Temp', 'Min_Temp', 'Max_Temp', 'Precipitation', 'Working_days', 'HA_Harvested']
            y_var = 'FFB_Yield'
            
            r2_list = []
            
            # Loop through all x_vars and run linear regression individually
            for var in x_vars:
                x = self.data[[var]]
                y = self.data[y_var]
                
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
                
                model = LinearRegression()
                model.fit(x_train, y_train)
                
                y_pred = model.predict(x_test)
                
                r2 = r2_score(y_test, y_pred)
                r2_list.append((var, abs(r2)))
                print('R2 result for {0} are {1}'.format(var, r2))
            
            # Sort the list of tuples based on R-squared values
            sorted_r2_list = sorted(r2_list, key=lambda x: x[1], reverse=True)
            
            # Select the top 3 x_vars
            best_2_x_vars = [var for var, r2 in sorted_r2_list[:2]]
            
            print('Best 2 x_vars based on individual R-squared values: ', best_2_x_vars)
            
            # Run linear regression for the best 3 x_vars and check the model performance
            x = self.data[best_2_x_vars]
            y = self.data[y_var]
            
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
            
            model = LinearRegression()
            model.fit(x_train, y_train)
            
            y_pred = model.predict(x_test)
            
            # Checking coefficients
            coeff_df = pd.DataFrame(model.coef_, best_2_x_vars, columns=['Coefficient'])
            print(coeff_df)
            
            # Checking model performance
            print('Training score for best 2 x_vars: ', model.score(x_train, y_train))
            print('Test score for best 2 x_vars: ', model.score(x_test, y_test))
            
            # Run linear regression for all x_vars and check the model performance
            x = self.data[x_vars]
            y = self.data[y_var]
            
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
            
            model_all_vars = LinearRegression()
            model_all_vars.fit(x_train, y_train)
            
            # Checking coefficients for all x_vars
            coeff_df_all_vars = pd.DataFrame(model_all_vars.coef_, x_vars, columns=['Coefficient'])
            print(coeff_df_all_vars)
            
            # Checking model performance for all x_vars
            print('Training score for all x_vars: ', model_all_vars.score(x_train, y_train))
            print('Test score for all x_vars: ', model_all_vars.score(x_test, y_test))
            
    def answer_compare_models(self):

        x_vars = ['SoilMoisture', 'Average_Temp', 'Min_Temp', 'Max_Temp', 'Precipitation', 'Working_days', 'HA_Harvested']
        y_var = 'FFB_Yield'
        
        # Run models for all x_vars and check the model performance
        x = self.data[x_vars]
        y = self.data[y_var]
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=0),
            'Gradient Boosting': GradientBoostingRegressor(random_state=0)
        }
        
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            
            # Checking model performance
            print(f'Model: {model_name}')
            print('Training score: ', model.score(x_train, y_train))
            print('Test score: ', r2_score(y_test, y_pred))
            print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
            print('-----------------------------------------')
        


class Q3:
    def __init__(self):
        self.text = '''As a term, data analytics predominantly refers to an assortment of applications, from basic business intelligence (BI), reporting and online analytical processing (OLAP) to various forms of advanced analytics. In that sense, it's similar in nature to business analytics, another umbrella term for approaches to analyzing data -- with the difference that the latter is oriented to business uses, while data analytics has a broader focus. The expansive view of the term isn't universal, though: In some cases, people use data analytics specifically to mean advanced analytics, treating BI as a separate category. Data analytics initiatives can help businesses increase revenues, improve operational efficiency, optimize marketing campaigns and customer service efforts, respond more quickly to emerging market trends and gain a competitive edge over rivals -- all with the ultimate goal of boosting business performance. Depending on the particular application, the data that's analyzed can consist of either historical records or new information that has been processed for real-time analytics uses. In addition, it can come from a mix of internal systems and external data sources. At a high level, data analytics methodologies include exploratory data analysis (EDA), which aims to find patterns and relationships in data, and confirmatory data analysis (CDA), which applies statistical techniques to determine whether hypotheses about a data set are true or false. EDA is often compared to detective work, while CDA is akin to the work of a judge or jury during a court trial -- a distinction first drawn by statistician John W. Tukey in his 1977 book Exploratory Data Analysis. Data analytics can also be separated into quantitative data analysis and qualitative data analysis. The former involves analysis of numerical data with quantifiable variables that can be compared or measured statistically. The qualitative approach is more interpretive -- it focuses on understanding the content of non-numerical data like text, images, audio and video, including common phrases, themes and points of view'''

    def a(self):
        
        lines = self.text.split('. ') # Split the text into lines
        self.lines = lines
        total_lines = len(lines) # Total number of lines
        lines_with_data = sum(['data' in line for line in lines]) # Number of lines where "data" occurs
        probability_data = lines_with_data / total_lines
        self.lines_with_data = lines_with_data
        
        print(f'Probability "Data" in each line is {probability_data}')
    
    def b(self):
        
        words = re.findall(r'\b\w+\b', self.text.lower())
        self.distinct_word_count = Counter(words)
        
        self.num_distinct_words = len(self.distinct_word_count)

        print(f'Distinct Word Count Distribution are: {self.distinct_word_count}')
        print(f'Total distinct words identified from the text are: {self.num_distinct_words}')
        
    def c(self):
        lines = self.lines
        lines_with_data = self.lines_with_data
        analytics_after_data = sum(['data analytics' in line for line in lines])
        probability_analytics_after_data = analytics_after_data / lines_with_data
        
        print(f"Probability of the word 'analytics' occurring after the word 'data': {probability_analytics_after_data}")
        
if __name__ == '__main__':
    print('''Question 1:
          A customer informed their consultant that they have developed several formulations of petrol that gives different characteristics of burning pattern. The formulations are obtaining by adding varying levels of additives that, for example, prevent engine knocking, gum prevention, stability in storage, and etc. However, a third party certification organisation would like to verify if the formulations are significantly different, and request for both physical and statistical proof. Since the formulations are confidential information, they are not named in the dataset. Please assist the consultant in the area of statistical analysis by doing this
          ''')
    #Initialize Q1
    q1_answer = Q1()
    print('''\na) A descriptive analysis of the additives (columns named as “a” to “i”), which must include summaries of findings (parametric/non-parametric). Correlation and ANOVA, if applicable, is a must''')
    q1_answer.a()
    print('Answer:')
    print(f'''The summary of additives databases are as below:
{q1_answer.summary}

To check for normality of the database, shapiro test are conducted.
Null Hypothesis: All data are normally distributed (Parametric)
Alternative Hypothesis: The data are not normally distributed (Non-Parametric)

The results are as follows:
    
    {q1_answer.normality_test_results}

As p for all additives are less than 0.05, the hypothesis is rejected thus the database is non-parametric.

The correlation matrix are then performed. As the data is non-parametric, ANOVA test are deemed invalid and only Spearman correlation test are conducted:
    
    {q1_answer.correlation_matrix}

  The result shows:
      Strong positive correlation:
          1) Between additives a and g with 0.7038.
          2) Between additives d and h with 0.4746
          3) Between addtives b and h with 0.4111
      
      Strong Negative correlation:
          1) Between additives b and f with -0.5845
          2) Between additives a and e with -0.5257
          3) Between additives c and d with -0.51242
      
      Very week correlation:
          1) Between e and f at -0.0072
          2) Between g and h at -0.0078
          3) Between i and h at 0.0097
            ''')
    print('\nb) A graphical analysis of the additives, including a distribution study')
    q1_answer.b()
    print('Answer:')
    print('''
The distribution plot for all additives can be summarized as follows:
     a) Right skewness tendency in additives a, b, d, f, g, h, i
     b) Left skewness tendency in additives c, e
 
The boxplot for all additives shows that additives a, c, d, f, h, i are in range of 0 to 10, while additives b and g from 10 to 20 and the additives e as the outliers with all its data is more than 70.
 
The correlation heatmap showing the correlation between additives with answer as per a.
            ''')
    print('\c) A clustering test of your choice (unsupervised learning), to determine the distinctive number of formulations present in the dataset.')
    q1_answer.c()
    print('Answer:')
    print(f'''
The elbow method shows that there are 4 main clusters to be plotted from the additives.
The mean value of each additives to be used in the clusters are as below:
    {q1_answer.data.groupby('cluster').mean()}
          ''')
    print('''\n\nQuestion 2:
A team of plantation planners are concerned about the yield of oil palm trees, which seems to fluctuate. They have collected a set of data and needed help in analysing on how external factors influence fresh fruit bunch (FFB) yield. Some experts are of opinion that the flowering of oil palm tree determines the FFB yield, and are linked to the external factors. Perform the analysis, which requires some study on the background of oil palm tree physiology. 
          ''')
    q2_answer = Q2()
    print('Answer: ')
    q2_answer.answer()
    print('''
The first test conducted to the external factors that may influence FFB yield are using the linear regression.

We start by identifying the correlation between each of the external factors to the FFB yield. With that, it was identified that 2 best correlation can be found via 'Max_Temp' and 'HA_Harversted'

The database are then split randomly into 80% train and 20% test data and linear regression models are being applied to:
    1) The best 2 external factors ('Max_Temp' & 'HA_Harvested')
    2) All the external factors

The results shows that the test score for the best 2 external factors are only 0.08 while for all are better at 0.24.

This can be interpreted that FFB Yield may be able to be predicted better utilizing all of the external factors given rather than selected few external factors even though it shows a better correlation with FFB yield.
          ''')
    q2_answer.answer_compare_models()
    print('''
The next test conducted to the database are to check the impact of different model in predicting the FFB yield.

3 model are chosen which are:
      1) Linear Regression
      2) Random Forest
      3) Gradient Boosting (xGBoost)

The results shows that the additional 2 model didnt improve the predictability of FFB Yield with Linear Regression still giving higher test score about 24% with lowest MSE.

Random Forest and Gradient Boosting did give a high training score with about 90% and 97% respectively, even so the test score are poor at 11% for random forest and -2% for Gradient Boosting.

In conclusion,the simple linear regression are deemed better in predicting the FFB yield with the external factors.

It was show via linear regression, external factors provided could impact the FFB yield with 24% predictability. 

Even so, additional factors could be included into account such as the impact of timing of FFB yield, soil type used and possibilities of bugs infesstation to the FFB yield.  
          ''')
    print('''\n\nQuestion 3:
Feed the following paragraph into your favourite data analytics tool, and answer the following;
    a. What is the probability of the word “data” occurring in each line ?
    b. What is the distribution of distinct word counts across all the lines ? 
    c. What is the probability of the word “analytics” occurring after the word “data” ?  
          ''')
    q3_answer = Q3()
    print('Answer: ')
    q3_answer.a()
    print()
    q3_answer.b()
    print()
    q3_answer.c()
