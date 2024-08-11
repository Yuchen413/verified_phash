import pandas as pd
import numpy as np

data = []
columns = ['file', 'optimized_file', 'l2', 'l_inf', 'ssim', 'steps', 'source']

with open('collision_test100/logs/preimage_attack.csv', 'r') as file:
    for line in file:
        line = line.strip()
        parts = line.split(',')
        if len(parts) == len(columns):
            data.append(parts)
        else:
            print("Skipped line:", line)

df = pd.DataFrame(data, columns=columns)
print(f'l2: {np.mean([float(i) for i in df.loc[:,"l2"].tolist()])}')
print(f'l_inf: {np.mean([float(i) for i in df.loc[:,"l_inf"].tolist()])}')
print(f'ssim: {np.mean([float(i) for i in df.loc[:,"ssim"].tolist()])}')
print(f'steps: {np.mean([float(i) for i in df.loc[:,"steps"].tolist()])}')

num_sample = 100

# bins = [1./255, 8./255, 16./255, 32./255, 64./255, 255./255]
# new_df = df.loc[:,"l_inf"].astype(float)
# new_df['Range'] = pd.cut(new_df, bins=bins)
# stats = new_df.groupby('Range').agg(['count'])
# # print(stats)

df['l_inf'] = df['l_inf'].astype(float)

# Define your bins
bins = [1./255, 8./255, 16./255, 32./255, 64./255, 255./255]
df['Range'] = pd.cut(df['l_inf'], bins=bins)
# stats = df.groupby('Range').agg(['count'])
grouped = df.groupby('Range').size()

# Calculate percentages
total_count = grouped.sum()
percentages = grouped / total_count * 100

# Create a new DataFrame to display results
results = pd.DataFrame({
    'Count': grouped,
    'Percentage': percentages.map('{:.2f}%'.format)
})

# Print results
print(results)
print(f'Overall ALL Collision Rate: {(len(df)/num_sample * 100)}%')








