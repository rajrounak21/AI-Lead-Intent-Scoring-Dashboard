import pandas as pd
import numpy as np
import random

np.random.seed(42)

def generate_phone_number():
    return "+91-" + ''.join([str(random.randint(0, 9)) for _ in range(10)])

def generate_email(name):
    domains = ['@gmail.com', '@yahoo.com', '@outlook.com', '@test.com']
    return name.lower().replace(" ", ".") + random.choice(domains)

def generate_comments():
    comment_pool = [
        "interested", "not interested", "call me now", "urgent", "need more info",
        "follow up next week", "budget approved", "not urgent", "price too high", ""
    ]
    return random.choice(comment_pool)

names = [f"User{i}" for i in range(10000)]
age_groups = ["18–25", "26–35", "36–50", "51+"]
family_status = ["Single", "Married", "Married with Kids"]
occupations = [
    "Software Engineer", "Teacher", "Doctor", "Lawyer", "Accountant"
    "Civil Engineer", "Sales Executive", "Business Owner", "Student", "Retired"
]

credit_scores = np.random.randint(300, 851, size=10000)
incomes = np.random.randint(100000, 1000001, size=10000)
lead_intent_balanced = np.concatenate([np.ones(5000, dtype=int), np.zeros(5000, dtype=int)])
np.random.shuffle(lead_intent_balanced)

df = pd.DataFrame({
    "Name": names,
    "Phone Number": [generate_phone_number() for _ in range(10000)],
    "Email": [generate_email(name) for name in names],
    "Credit Score": credit_scores,
    "Age Group": np.random.choice(age_groups, size=10000),
    "Family Background": np.random.choice(family_status, size=10000),
    "Income": incomes,
    "Comments": [generate_comments() for _ in range(10000)],
    "Occupation": np.random.choice(occupations, size=10000),
    "Lead Intent": lead_intent_balanced
})

# Save to CSV
df.to_csv("lead.csv", index=False)
print("✅ Dataset saved as lead.csv")
