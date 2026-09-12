import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title='Sales Prediction', page_icon='📊', layout='wide')

@st.cache_data
def load_data():
    df = pd.read_csv('Advertising.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df.drop_duplicates().dropna()

df = load_data()
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.sidebar.title('📌 Navigation')
page = st.sidebar.radio('Select Page', ['🏠 Home', '📊 Data Analysis', '📈 Visualizations', '🤖 Model Performance', '🔮 Sales Prediction', '💡 Business Insights'])

if page == '🏠 Home':
    st.title('📊 Sales Prediction using Python')
    st.markdown('''### Machine Learning Sales Prediction Dashboard\nPredict **Sales** using advertising expenditure on **TV, Radio and Newspaper**.\n\nThe project uses **Linear Regression**.''')
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Total Records',len(df)); c2.metric('Features',3); c3.metric('Average Sales',f'{y.mean():.2f}'); c4.metric('Model R²',f'{r2:.3f}')
    st.subheader('📋 Dataset Preview'); st.dataframe(df.head(10), use_container_width=True)

elif page == '📊 Data Analysis':
    st.title('📊 Data Analysis')
    c1,c2=st.columns(2)
    with c1:
        st.write('Rows:',df.shape[0]); st.write('Columns:',df.shape[1]); st.write('Column names:',list(df.columns))
    with c2: st.write('Missing values:'); st.write(df.isnull().sum())
    st.subheader('📈 Statistical Summary'); st.dataframe(df.describe().round(2), use_container_width=True)
    st.subheader('🔗 Correlation with Sales'); st.dataframe(df.corr(numeric_only=True)['Sales'].sort_values(ascending=False).to_frame('Correlation'), use_container_width=True)

elif page == '📈 Visualizations':
    st.title('📈 Advertising & Sales Visualizations')
    for channel in ['TV','Radio','Newspaper']:
        fig,ax=plt.subplots(figsize=(9,4)); sns.scatterplot(data=df,x=channel,y='Sales',ax=ax); ax.set_title(f'{channel} Advertising vs Sales'); st.pyplot(fig)
    fig,ax=plt.subplots(figsize=(8,6)); sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm',fmt='.2f',ax=ax); ax.set_title('Correlation Matrix'); st.pyplot(fig)

elif page == '🤖 Model Performance':
    st.title('🤖 Linear Regression Model')
    c1,c2,c3,c4=st.columns(4); c1.metric('MAE',f'{mae:.3f}'); c2.metric('MSE',f'{mse:.3f}'); c3.metric('RMSE',f'{rmse:.3f}'); c4.metric('R² Score',f'{r2:.3f}')
    st.subheader('Model Coefficients')
    coef=pd.DataFrame({'Advertising Channel':X.columns,'Coefficient':model.coef_}); st.dataframe(coef.round(4),use_container_width=True,hide_index=True); st.write(f'Model Intercept: {model.intercept_:.4f}')
    st.subheader('Actual vs Predicted Sales')
    fig,ax=plt.subplots(figsize=(8,6)); sns.scatterplot(x=y_test,y=y_pred,ax=ax); mn=min(y_test.min(),y_pred.min()); mx=max(y_test.max(),y_pred.max()); ax.plot([mn,mx],[mn,mx],linestyle='--'); ax.set_xlabel('Actual Sales'); ax.set_ylabel('Predicted Sales'); st.pyplot(fig)

elif page == '🔮 Sales Prediction':
    st.title('🔮 Predict Future Sales')
    c1,c2,c3=st.columns(3)
    with c1: tv=st.number_input('📺 TV Advertising',min_value=0.0,value=200.0,step=10.0,key='tv')
    with c2: radio=st.number_input('📻 Radio Advertising',min_value=0.0,value=40.0,step=5.0,key='radio')
    with c3: newspaper=st.number_input('📰 Newspaper Advertising',min_value=0.0,value=30.0,step=5.0,key='newspaper')
    if st.button('🚀 Predict Sales',type='primary'):
        inp=pd.DataFrame({'TV':[tv],'Radio':[radio],'Newspaper':[newspaper]})
        prediction=model.predict(inp)[0]
        st.success(f'### Predicted Sales: {prediction:.2f}')
        st.subheader('Advertising Budget'); st.bar_chart(pd.DataFrame({'Channel':['TV','Radio','Newspaper'],'Advertising Spend':[tv,radio,newspaper]}).set_index('Channel'))

elif page == '💡 Business Insights':
    st.title('💡 Business Insights')
    coef=pd.DataFrame({'Channel':X.columns,'Coefficient':model.coef_})
    strongest=coef.loc[coef['Coefficient'].abs().idxmax(),'Channel']
    st.write(f'🔹 **{strongest}** has the largest absolute regression coefficient.')
    st.write(f'🔹 The model explains approximately **{r2*100:.2f}%** of the variation in Sales on the test data.')
    st.write(f'🔹 Average Sales is **{y.mean():.2f}**.')
    st.markdown('''### Marketing Recommendations\n1. Focus attention on channels with stronger modeled relationships with Sales.\n2. Test different budgets with the prediction page.\n3. Compare predicted sales with actual campaign results.\n4. Consider ROI, customer behavior and market conditions before making final decisions.''')
    st.warning('Regression shows statistical association; it does not by itself prove causation.')

st.divider(); st.caption('Sales Prediction using Python | Linear Regression | Streamlit')
