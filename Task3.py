import pandas as pd
import matplotlib.pyplot as plt
#reading csv file
file_path=('C:\\Users\\CHITTURI SRINIVAS\\Downloads\\householdtask3.csv')
data=pd.read_csv(file_path)
data.columns=data.columns.str.strip()
print(data.head(10))

#printing column names
print(data.columns)

#SACTTER PLOT
plt.figure(figsize=(10,5))
plt.scatter(data['year'],data['own'])
#title of the plot
plt.title("SCATTER PLOT")
#LABELLING THE AXIS
plt.xlabel('Year')
plt.ylabel('Own')
#PLOT
plt.show()
#plot 2
plt.figure(figsize=(10,5))
plt.scatter(data['year'],data['own'])
plt.title("SCATTER PLOT WITH COLOR BAR")
#LABELLING THE AXIS
plt.xlabel('Year')
plt.ylabel('Own')
plt.colorbar()
plt.show()

plt.plot(data['year'])
plt.plot(data['own'])
plt.title("SCATTER PLOT")
#LABELLING THE AXIS
plt.xlabel('Year')
plt.ylabel('Own')
plt.show()
plt.figure(figsize=(8, 5))
df=data.head(10)
# Plotting with corrected column name
plt.bar(df['age'], df['income'], color='skyblue', label='income')

# bargraph plot
# Add labels and title
plt.xlabel('age')
plt.ylabel('income')
plt.title('Bar Chart of age vs. income')

# Add legend
plt.legend()

# Rotate x-axis labels if necessary
plt.xticks(rotation=45)

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Show plot
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(data['eqv_income'], data['eqv_exp'], marker='o', color='skyblue', linestyle='-', label='eqv_exp')

# Add labels and title
plt.title('eqv_exp Over eqv_income')
plt.xlabel('eqv_income')
plt.ylabel('eqv_exp')

# Add legend
plt.legend()

# Add gridlines
plt.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


