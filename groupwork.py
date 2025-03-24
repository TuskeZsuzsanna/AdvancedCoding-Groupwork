#groupwork

#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import stats

#-----------Input data-----------
creditdata = pd.read_csv("train_biased.csv")

creditdata.head(30)
creditdata.info()
description = creditdata.describe()



#-----------Analyzing the data before data cleansing-----------
#distribution of numeric data

def distribution_hist(data, dataname):
    plt.hist(data, color="mediumaquamarine")
    plt.title("Distribution of "+dataname)
    plt.xlabel(dataname)
    plt.ylabel("Frequency")
    plt.show()

distribution_hist(creditdata.Age, "Age")    
distribution_hist(creditdata.Monthly_Inhand_Salary, "Monthly Inhand Salary")
distribution_hist(creditdata.Num_Bank_Accounts, "Number of bank accounts")
distribution_hist(creditdata.Num_Credit_Card, "Number of credit cards")
distribution_hist(creditdata.Interest_Rate, "Interest Rate")
distribution_hist(creditdata.Delay_from_due_date, "Delay from due date")
distribution_hist(creditdata.Num_Credit_Inquiries, "Number of credit inquires")
distribution_hist(creditdata.Credit_Utilization_Ratio, "Credit utilization ratio")
distribution_hist(creditdata.Credit_Score, "Credit score")


#Frequency of cathegoric data
def cathegoric_bar(data, columnname, dataname):
    pivot = data.groupby(data[columnname]).agg(
        Number=(columnname, "count"))
    
    pivot = pivot.reset_index()
    
    plt.bar(pivot[columnname], pivot.Number, color="mediumaquamarine")
    plt.title("Frequency of "+dataname)
    plt.xlabel(dataname)
    plt.xticks(rotation = 90)
    plt.ylabel("Frequency")
    plt.show()

cathegoric_bar(creditdata.loc[:, ["City"]], "City", "Cities")  
cathegoric_bar(creditdata.loc[:, ["Month"]], "Month", "Months")
cathegoric_bar(creditdata.loc[:, ["Occupation"]], "Occupation", "Occupations")
#cathegoric_bar(creditdata.loc[:, ["Type_of_Loan"]], "Type_of_Loan", "Types of Loans") #too many variations, not worth visualizing
cathegoric_bar(creditdata.loc[:, ["Credit_Mix"]], "Credit_Mix", "Credit mixes")
cathegoric_bar(creditdata.loc[:, ["Payment_of_Min_Amount"]], "Payment_of_Min_Amount", "Existence of Minimum Payment Amount")



#connection of numeric data to the credit score
def connection_to_creditscore(data, dataname):
    valid_data = ~data.isna() & ~creditdata.Credit_Score.isna()
    x_clean = data[valid_data]
    y_clean = creditdata.Credit_Score[valid_data]

    coeffs = np.polyfit(x_clean, y_clean, 1)  # Fit trendline
    trendline = np.poly1d(coeffs)  # Create polynomial function

    plt.scatter(x_clean, y_clean, label="Data", color='mediumslateblue')
    plt.plot(x_clean, trendline(x_clean), color='lightcoral', label="Trendline")
    
    plt.xlabel(dataname)
    plt.ylabel("Credit Score")
    plt.legend()
    plt.title("Connection between credit score and "+dataname)
    plt.show()
       
connection_to_creditscore(creditdata.Age, "Age")    
connection_to_creditscore(creditdata.Monthly_Inhand_Salary, "Monthly Inhand Salary")
connection_to_creditscore(creditdata.Num_Bank_Accounts, "Number of bank accounts")
connection_to_creditscore(creditdata.Num_Credit_Card, "Number of credit cards")
connection_to_creditscore(creditdata.Interest_Rate, "Interest Rate")
connection_to_creditscore(creditdata.Delay_from_due_date, "Delay from due date")
connection_to_creditscore(creditdata.Num_Credit_Inquiries, "Number of credit inquires")
connection_to_creditscore(creditdata.Credit_Utilization_Ratio, "Credit utilization ratio")



#connection between cathegoric values and credit score
def cathegoric_to_creditscore(data, columnname, dataname):
    pivot = data.groupby(data[columnname]).agg(
        Mean=("Credit_Score", np.mean))   
    pivot = pivot.reset_index()
    
    plt.bar(pivot[columnname], pivot.Mean, color='mediumslateblue')
    plt.title("Average credit score by "+dataname)
    plt.xlabel(dataname)
    plt.xticks(rotation = 90)
    plt.ylabel("Average credit score")
    plt.show()

cathegoric_to_creditscore(creditdata.loc[:, ["Credit_Score", "City"]], "City", "Cities")  
cathegoric_to_creditscore(creditdata.loc[:, ["Credit_Score", "Month"]], "Month", "Months")
cathegoric_to_creditscore(creditdata.loc[:, ["Credit_Score", "Occupation"]], "Occupation", "Occupations")
#cathegoric_to_creditscore(creditdata.loc[:, ["Credit_Score", "Type_of_Loan"]], "Type_of_Loan", "Types of Loans") #too many variations, not worth visualizing
cathegoric_to_creditscore(creditdata.loc[:, ["Credit_Score", "Credit_Mix"]], "Credit_Mix", "Credit mixes")
cathegoric_to_creditscore(creditdata.loc[:, ["Credit_Score", "Payment_of_Min_Amount"]], "Payment_of_Min_Amount", "Existence of Minimum Payment Amount")



#-----------Data cleaning-----------
#__________step 1__________
#handling customer IDs and names
NameCustomerDict = creditdata.loc[:, ["Name", "Customer_ID"]]
NameCustomerDict = NameCustomerDict.drop_duplicates(keep='first') #every name-customerID should appear once
NameCustomerDict = NameCustomerDict.dropna() #lets drop the NAs

#___step 1a___
#counting how many names belong to one customerID
NameCustomerDicControll2 = NameCustomerDict.groupby(NameCustomerDict.Customer_ID).agg(
        Num = ("Name", "count"))
NameCustomerDicControll2 = NameCustomerDicControll2.reset_index()
NameCustomerDicControll2[NameCustomerDicControll2.Num > 1]
#to every customer ID only belongs one name

#filling up the missing names with the help of customer IDs
creditdata = creditdata.merge(NameCustomerDict[['Customer_ID', 'Name']], on="Customer_ID", how="left")
creditdata.Name_x = np.where(creditdata.Name_x.isna(), creditdata.Name_y, creditdata.Name_x)
creditdata = creditdata.drop(columns=['Name_y']) #drop the Name_y column


#___step 1b___
#counting how many customerID belong to one name
NameCustomerDicControll = NameCustomerDict.groupby(NameCustomerDict.Name).agg(
        Num = ("Customer_ID", "count"))
NameCustomerDicControll = NameCustomerDicControll.reset_index()
NameCustomerDicControll[NameCustomerDicControll.Num > 1].count()
#in 1823 cases we cannot match the customer ID to the given name


#if this number is just 1, then we can fill the NA values with the real values
NameCustomerDict = NameCustomerDict[NameCustomerDict.Name.isin(NameCustomerDicControll.loc[NameCustomerDicControll.Num == 1, "Name"])]
creditdata = creditdata.merge(NameCustomerDict, left_on= "Name_x", right_on = "Name", how="left")
creditdata.Customer_ID_x = np.where(creditdata.Customer_ID_x.isna(), creditdata.Customer_ID_y, creditdata.Customer_ID_x )
creditdata = creditdata.drop(columns=['Name', 'Customer_ID_y'])

#rename the columns to the original names
creditdata = creditdata.rename(columns={"Customer_ID_x": "Customer_ID", "Name_x": "Name"})
creditdata.count()

problematic = creditdata[creditdata.Name.isna() | creditdata.Customer_ID.isna()]


creditdata = creditdata.rename(columns={"Person_x": "Person"})
creditdata = creditdata.rename(columns={"Customer_ID_x": "Customer_ID"})

#__________step 2__________
#there is always the same person 8 times after each other
#this person usually should have the same personal data

#__________step 2a__________
#creating a new column that gives one number for one customer
creditdata["customer_index"] = np.floor(creditdata.index / 8)
Person_Customerindex_Dictionary = creditdata[["Name", "Customer_ID", "customer_index"]].dropna().drop_duplicates(keep = "first")
#perfect, there are 12500 rows in the Person_Customerindex_Dictionary, which means,
#that we have the names and the customer_IDs of every customer, because 100,000 / 8 = 12,500

#filling up data according to the customer index - function
def customer_index_filler(creditdata, dictionary, dataname):
    dictionary = dictionary.dropna().drop_duplicates(keep = "first")
    if pd.Series(dictionary.customer_index).nunique() == 12500 and dictionary.customer_index.count() == 12500:
        creditdata = creditdata.merge(dictionary, on = "customer_index", how = "left")
        creditdata[dataname + "_x"] = np.where(creditdata[dataname + "_x"].isna(), creditdata[dataname + "_y"], creditdata[dataname + "_x"])
        print(creditdata[creditdata[dataname + "_x"] != creditdata[dataname + "_y"]]) #if something is here, we filled up the data wrong
        creditdata = creditdata.iloc[: ,0: (len(creditdata.columns) -1)]
        creditdata = creditdata.rename(columns={dataname + "_x": dataname})
    else:
        print("The dictionary is not correct")
    return creditdata


#__________step 2b__________
#filling up columns with the help of the customer index

#__Name__
creditdata = customer_index_filler(creditdata, creditdata[["Name", "customer_index"]], "Name")

#__Customer_ID__
creditdata = customer_index_filler(creditdata, creditdata[["Customer_ID", "customer_index"]], "Customer_ID")

#__SSN__
#changing wrong SSNs to nan, and then running the customer_index_filler function for that too
#format of a SSN should be 3 numbers - 2 numbers - 4 numbers
creditdata.SSN = np.where(
    creditdata.SSN.apply(lambda x: bool(re.fullmatch(r"\d{3}-\d{2}-\d{4}", str(x)))),
    creditdata.SSN,
    np.nan)
creditdata = customer_index_filler(creditdata, creditdata[["SSN", "customer_index"]], "SSN")

#__Occupation__
creditdata = customer_index_filler(creditdata, creditdata[["Occupation", "customer_index"]], "Occupation")


#__City + Street__
#observing if anybody has moved
Moving = creditdata[["customer_index", "City", "Street"]].dropna().drop_duplicates(keep = "first")

Moving_Pivot = Moving.groupby("customer_index").agg(
    NumOfStreets = ("Street", "nunique"),
    NumOfCities = ("City", "nunique"))

problematic = Moving_Pivot[(Moving_Pivot.NumOfStreets != Moving_Pivot.NumOfCities)]
#everybody has the same street name, but sometimes different cities
#it is a very low chance, that someone has moved and their street name is the same, i think we just have wrong values

creditdata = customer_index_filler(creditdata, creditdata[["Street", "customer_index"]], "Street")

#putting the most used city into a city dictionary
CityDictionary = creditdata[["customer_index", "City", "Street"]].dropna().groupby("customer_index").agg(
    City = ("City", lambda x: x.mode().iloc[0])).reset_index()

creditdata = customer_index_filler(creditdata, CityDictionary, "City") 


#__________step 3__________
#creating a new month column and filling up with the months from january to august,
#since there is a pattern, that for every person there are these 8 months after each other in this order in the dataframe
creditdata["Month2"] = 1
MonthDict = np.array(["January", "February", "March", "April", "May", "June", "July", "August"])

for i in creditdata.index:
    creditdata.loc[i, "Month2"] = MonthDict[i%8]


problematic = creditdata.loc[creditdata.Month != creditdata.Month2, "Month"].dropna()
creditdata.Month = creditdata.Month2
creditdata = creditdata.drop(columns=["Month2"])




#__________step 4__________

#__Age__
#deleting negative and very big numbers
Ages = np.sort(creditdata.Age.unique())
creditdata.loc[creditdata.Age < 0, "Age"] = np.nan
creditdata.loc[creditdata.Age > 85, "Age"] = np.nan



AgeDict = creditdata[["customer_index", "Age"]].dropna().drop_duplicates(keep ="first" )

#what is the minimum, maximum age that belongs to the same person
AgeDictControll = AgeDict.groupby("customer_index").agg(
    Number = ("customer_index", "count"),
    Min = ("Age", "min"),
    Max = ("Age", "max"))
AgeDictControll  = AgeDictControll.reset_index()
AgeDictControll["Range"] = AgeDictControll.Max - AgeDictControll.Min
AgeDictControll["Substitute"] = (AgeDictControll.Max + AgeDictControll.Min)/2

#some people had their birthday during the time the data was recorded, we need to handle this
#but there are no wrong values in the age field apart from these
print(AgeDictControll[AgeDictControll.Range > 1 ])


creditdata = creditdata.merge(AgeDictControll, on="customer_index", how="left")
#to delete
Ages = creditdata.Age
creditdata.Age = Ages

creditdata["Age2"] = creditdata.Age
creditdata["Age2"] = np.where(creditdata["Age2"].notna(),creditdata["Age2"],
                             np.where(creditdata.Month == "January", creditdata.Min,
                                      np.where(creditdata.Month == "August", creditdata.Max,
                                               np.where(creditdata["Age2"].shift(-1) == creditdata["Age2"].shift(1), creditdata["Age2"].shift(1),
                                                        np.where((creditdata["Age2"].shift(-1).isna() == False) & (creditdata["Age2"].shift(1).isna() == False), creditdata.Substitute,
                                                                 np.where((creditdata["Age2"].shift(1).isna() == False) & (creditdata["Age2"].shift(-1).isna() == True),creditdata["Age2"].shift(1), creditdata["Age2"].shift(-1)                                                                          
                             ))))))

creditdata[creditdata.Age2.isna()]
problematic = creditdata[["Month", "Name", "Age", "Age2"]]
problematic[problematic.isna()]



#there are 362 rows, for which this code for the age columns cleaning doesnt work, i will come back and fix it, until that, please use the following row
creditdata["Age2"] = np.where(creditdata["Age2"].notna(),creditdata["Age2"],creditdata.Substitute)


creditdata["Age"] = creditdata["Age2"]
creditdata = creditdata.iloc[:, :28]
  
creditdata.count()



#__________step 5__________
#cleaning numeric data

#__________step 5a__________
#function for removing _ characters from the end of a line
def character_cleaner(column):
    return column.astype(str).apply(lambda x: x[:-1] if x.endswith("_") else x).astype(float)
    
creditdata.Annual_Income = character_cleaner(creditdata.Annual_Income)
creditdata.Num_of_Loan = character_cleaner(creditdata.Num_of_Loan)


#__________step 5b__________
#using the customer_index_filler function where its possible
#if not possible, then we should remove the "very outlier" values

creditdata.Annual_Income.describe()
creditdata = customer_index_filler(creditdata, creditdata[["customer_index", "Annual_Income"]], "Annual_Income")




#__________step 5c__________
#turning the age codes into a function and using it on several columns, where a value can change











