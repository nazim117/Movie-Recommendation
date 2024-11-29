import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(
    'data/name.basics.tsv/name.basics.tsv',
    sep='\t',
    on_bad_lines='skip'
)

data['birthYear'] = pd.to_numeric(data['birthYear'], errors='coerce')
data['deathYear'] = pd.to_numeric(data['deathYear'], errors='coerce')

numeric_data = data[['birthYear', 'deathYear']].dropna()

sns.pairplot(numeric_data)
plt.show()
