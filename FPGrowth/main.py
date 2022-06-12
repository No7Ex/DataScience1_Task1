#############################################
####The following code was made by following an already implemented code, which can be found here: https://hands-on.cloud/implementation-of-fp-growth-algorithm-using-python/#:~:text=Related%20articles-,FP%2Dgrowth%20algorithm%20overview,information%20between%20the%20frequent%20items.
####After initial implementation it was adjusted to confirm to the task given. (streamlit implementation, slider implementation, etc.)
#############################################

# importing module
import pandas as pd
import numpy as np
import streamlit as st
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

#full page instead of central column
#st.set_page_config(layout="wide")



st.write("<h1>Dataset: Postoperative Patient Data</h1>", unsafe_allow_html=True)
st.write("Data was provided by Sharon Summers, School of Nursing, University of Kansas Medical Center, Kansas City, KS 66160 and "
         "Linda Woolery, School of Nursing, University of Missouri, Columbia, MO 65211 in 1993")
st.write("The aim is to determine where patients in a postoperative area should be sent to next.")
st.write(pd.read_csv("post-operative.csv"))
st.write('''Attribute Information:
  \n  1. L-CORE (patient's internal temperature in C):  \n      - lc-high (> 37), lc-mid (>= 36 and <= 37), lc-low (< 36)
  \n  2. L-SURF (patient's surface temperature in C):  \n      - ls-high (> 36.5), ls-mid (>= 36.5 and <= 35), ls-low (< 35)
  \n  3. L-O2 (oxygen saturation in %):  \n      - excellent (>= 98), good (>= 90 and < 98), fair (>= 80 and < 90), poor (< 80)
  \n  4. L-BP (last measurement of blood pressure):  \n      - bp-high (> 130/90), bp-mid (<= 130/90 and >= 90/70), bp-low (< 90/70)
  \n  5. SURF-STBL (stability of patient's surface temperature):  \n      - s-stable, s-mod-stable, s-unstable
  \n  6. CORE-STBL (stability of patient's core temperature)  \n      - c-stable, c-mod-stable, c-unstable
  \n  7. BP-STBL (stability of patient's blood pressure)  \n      - b-stable, b-mod-stable, b-unstable
  \n  8. COMFORT (patient's perceived comfort at discharge, measured as an integer between 0 and 20)
  \n  9. decision ADM-DECS (discharge decision):  \n      - I (patient sent to Intensive Care Unit),  \n      - S (patient prepared to go home),  \n      - A (patient sent to general hospital floor)''')


st.write("<h1>Variables & Sliders</h1>", unsafe_allow_html=True)
#Support
st.write('''<b>Support:</b> Percentage of item combinations in the database that contain both itemsets <b>X,Y:</b>''', unsafe_allow_html=True)
st.latex(r'''S(X \Rightarrow Y) = S(X \cup Y) = P (X \cup Y) ''')
#Confidence
st.write('''<b>Confidence:</b> Percentage of item combinations in the database D containing both itemsets <b>X,Y:</b>''', unsafe_allow_html=True)
st.latex(r''' C (X \Rightarrow Y) = P(Y|X) = \frac{S(X \cup Y )}{S(X)}''')
#Lift
st.write('''<b>Lift:</b> Measures frequency of X and Y, if both are statistically independent:''', unsafe_allow_html=True)
st.latex(r''' L(X \Rightarrow Y) = \frac{C(X \Rightarrow Y )}{S(Y)}''')
#Conviction
st.write('''<b>Conviction:</b> Measures implication strength of the rule from statistical independence:''', unsafe_allow_html=True)
st.latex(r''' Con(X \Rightarrow Y) = \frac{1- S(Y)}{1- C(X \Rightarrow Y)}''')



# dataset
dataset = pd.read_csv("post-operative.csv")

# Gather All Items of Each Transactions into Numpy Array
transaction = []
for i in range(0, dataset.shape[0]):
    for j in range(0, dataset.shape[1]):
        transaction.append(dataset.values[i,j])

# converting to numpy array
transaction = np.array(transaction)

#  Transform Them a Pandas DataFrame
df = pd.DataFrame(transaction, columns=["items"])

# Put 1 to Each Item For Making Countable Table, to be able to perform Group By
df["incident_count"] = 1

#  Delete NaN Items from Dataset
indexNames = df[df['items'] == "nan" ].index
df.drop(indexNames , inplace=True)

# Making a New Appropriate Pandas DataFrame for Visualizations
df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()

#  Initial Visualizations
# Transform Every Transaction to Seperate List & Gather Them into Numpy Array
transaction = []
for i in range(dataset.shape[0]):
    transaction.append([str(dataset.values[i,j]) for j in range(dataset.shape[1])])

# creating the numpy array of the transactions
transaction = np.array(transaction)

# initializing the transactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transaction).transform(transaction)
dataset = pd.DataFrame(te_ary, columns=te.columns_)

#sliders
support = st.slider("Support Value", min_value=0.1, max_value=0.8, value=0.65)
confidence = st.slider("Confidence Value", min_value=0.01, max_value=0.95, value=0.8)

# creating asssociation rules
res=fpgrowth(dataset,min_support=support, use_colnames=True)
res=association_rules(res, metric="confidence", min_threshold=confidence)

#st layout adjustments
res["antecedents"] = res["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
res["consequents"] = res["consequents"].apply(lambda x: list(x)[0]).astype("unicode")

st.write(res)



