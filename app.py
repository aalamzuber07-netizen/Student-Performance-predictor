import pandas as pd
import sklearn
from ucimlrepo import fetch_ucirepo

print("--- Health Check ---")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

# Fetch the dataset
dataset = fetch_ucirepo(id=320) 

# Create the data table (df) from the fetched data
X_full = dataset.data.features 
y_full = dataset.data.targets 
df = pd.concat([X_full, y_full], axis=1)

# Now your 'clues' code will work!
clues = ['G1', 'G2', 'absences', 'studytime', 'failures']
X = df[clues]
y = df['G3']

print("--------------------")
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

# 1. Fetch the data from the internet
dataset = fetch_ucirepo(id=320) 

# 2. Get the clues (features) and the answers (targets)
X_full = dataset.data.features 
y_full = dataset.data.targets 

# 3. Combine them into one table called 'df'
df = pd.concat([X_full, y_full], axis=1)

# 4. Pick the specific columns for our counselor tool
clues = ['G1', 'G2', 'absences', 'studytime', 'failures']
X = df[clues]
y = df['G3']

# 5. Split the data into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Success! The robot is ready to study {len(X_train)} students.")
from sklearn.linear_model import LinearRegression

# 1. Create the model "object"
model = LinearRegression()

# 2. The "Teaching" step (Fitting)
# We give it the training clues and the training answers
model.fit(X_train, y_train)

print("The robot has finished studying!")

# 3. Let's see what the robot learned!
# We'll look at the 'Importance' of each clue
importance = pd.Series(model.coef_, index=clues)
print("\n--- What the robot values most ---")
print(importance)
# 4. The Final Exam
# We ask the robot to predict grades for the students it hasn't seen yet
y_pred = model.predict(X_test)

# 5. How close was the robot?
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(y_test, y_pred)
print(f"\nOn average, the robot's guess was off by: {error:.2f} points.")
# 1. Grab the first student from our test set to analyze
student_index = 0 
sample_student = X_test.iloc[[student_index]]
actual_grade = y_test.iloc[student_index]
predicted_grade = y_pred[student_index]

# 2. Generate the report
print(f"\n--- Counselor Report for Student ---")
print(f"Predicted Grade: {predicted_grade:.1f}/20")
print(f"Actual Grade: {actual_grade}/20")

# 3. Actionable Insights Logic
if predicted_grade < 10:
    print("Action: Student is at risk of failing. Schedule a check-in.")
    if sample_student['absences'].values[0] > 10:
        print("Insight: High absences detected. Investigate barriers to attendance. 🚩")
    if sample_student['studytime'].values[0] < 2:
        print("Insight: Low study time. Recommend time-management coaching. 📖")
else:
    print("Action: Student is on track. Keep encouraging their current habits! ✨")