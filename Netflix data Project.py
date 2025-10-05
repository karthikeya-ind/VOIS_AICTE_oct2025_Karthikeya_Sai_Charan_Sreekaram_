import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
import numpy as np
import caas_jupyter_tools as tools

file_path = "/mnt/data/Netflix Dataset.csv"
df = pd.read_csv(file_path)

df.columns = [c.strip() for c in df.columns]
df['Release_Date_parsed'] = pd.to_datetime(df['Release_Date'], errors='coerce')
df['Release_Year'] = df['Release_Date_parsed'].dt.year
df['Category'] = df['Category'].str.strip()
df['Type'] = df['Type'].astype(str).str.strip()

def parse_duration(row):
    d = str(row['Duration']).strip()
    if pd.isna(d) or d == 'nan':
        return np.nan, None
    m = re.search(r'(\d+)\s*min', d, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), 'min'
    s = re.search(r'(\d+)\s*Season', d, flags=re.IGNORECASE)
    if s:
        return int(s.group(1)), 'season'
    n = re.search(r'(\d+)', d)
    if n:
        return int(n.group(1)), 'unknown'
    return np.nan, None

parsed = df.apply(parse_duration, axis=1, result_type='expand')
df['Duration_Value'] = parsed[0]
df['Duration_Unit'] = parsed[1]

dist = df['Category'].value_counts().rename_axis('Category').reset_index(name='Count')
tools.display_dataframe_to_user("Movies_vs_TVShows_Distribution", dist)
plt.figure(figsize=(6,4))
plt.bar(dist['Category'], dist['Count'])
plt.title('Distribution: Movies vs TV Shows')
plt.xlabel('Category')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

def split_countries(x):
    if pd.isna(x):
        return []
    parts = [p.strip() for p in str(x).split(',') if p.strip()]
    return parts

all_countries = df['Country'].dropna().apply(split_countries)
country_list = [c for sub in all_countries for c in sub]
country_counts = Counter(country_list)
top_countries = pd.DataFrame(country_counts.most_common(20), columns=['Country','Count'])
tools.display_dataframe_to_user("Top_20_Countries_by_Content_Count", top_countries)
plt.figure(figsize=(10,5))
plt.bar(top_countries['Country'], top_countries['Count'])
plt.title('Top 20 Countries by Content Count')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

def split_types(x):
    if pd.isna(x):
        return []
    parts = [p.strip() for p in str(x).split(',') if p.strip()]
    return parts

all_types = df['Type'].apply(split_types)
type_list = [t for sub in all_types for t in sub]
type_counts = Counter(type_list)
top_types = pd.DataFrame(type_counts.most_common(30), columns=['Genre','Count'])
tools.display_dataframe_to_user("Top_30_Genres", top_types)
plt.figure(figsize=(10,6))
plt.bar(top_types['Genre'], top_types['Count'])
plt.title('Top Genres in Dataset (Top 30)')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

year_counts = df['Release_Year'].dropna().astype(int).value_counts().sort_index().rename_axis('Year').reset_index(name='Count')
tools.display_dataframe_to_user("Content_by_Release_Year", year_counts.tail(50))
plt.figure(figsize=(10,5))
plt.plot(year_counts['Year'], year_counts['Count'], marker='o')
plt.title('Content Added by Release Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

movies = df[df['Category'].str.lower() == 'movie'].copy()
movies_min = movies[movies['Duration_Unit'] == 'min'].copy()
movies_min['Duration_Value'] = pd.to_numeric(movies_min['Duration_Value'], errors='coerce')

longest = movies_min.sort_values('Duration_Value', ascending=False).head(10)[['Title','Director','Country','Release_Year','Duration','Duration_Value']]
shortest = movies_min.sort_values('Duration_Value', ascending=True).head(10)[['Title','Director','Country','Release_Year','Duration','Duration_Value']]

tools.display_dataframe_to_user("Top_10_Longest_Movies", longest)
tools.display_dataframe_to_user("Top_10_Shortest_Movies", shortest)

tvshows = df[df['Category'].str.lower() == 'tv show'].copy()
tvshows_seasons = tvshows[tvshows['Duration_Unit'] == 'season'].copy()
tvshows_seasons['Duration_Value'] = pd.to_numeric(tvshows_seasons['Duration_Value'], errors='coerce')
longest_tv = tvshows_seasons.sort_values('Duration_Value', ascending=False).head(10)[['Title','Director','Country','Release_Year','Duration','Duration_Value']]
tools.display_dataframe_to_user("Top_10_TVShows_by_Seasons", longest_tv)

duration_stats = movies_min['Duration_Value'].describe().to_frame().reset_index().rename(columns={'index':'Metric', 'Duration_Value':'Value'})
tools.display_dataframe_to_user("Movie_Duration_Statistics", duration_stats)

out_paths = {}
longest.to_csv('/mnt/data/top_10_longest_movies.csv', index=False)
shortest.to_csv('/mnt/data/top_10_shortest_movies.csv', index=False)
top_countries.to_csv('/mnt/data/top_20_countries.csv', index=False)
top_types.to_csv('/mnt/data/top_30_genres.csv', index=False)
year_counts.to_csv('/mnt/data/content_by_release_year.csv', index=False)
dist.to_csv('/mnt/data/movies_tv_distribution.csv', index=False)
out_paths['top_10_longest_movies'] = '/mnt/data/top_10_longest_movies.csv'
out_paths['top_10_shortest_movies'] = '/mnt/data/top_10_shortest_movies.csv'
out_paths['top_20_countries'] = '/mnt/data/top_20_countries.csv'
out_paths['top_30_genres'] = '/mnt/data/top_30_genres.csv'
out_paths['content_by_release_year'] = '/mnt/data/content_by_release_year.csv'
out_paths['movies_tv_distribution'] = '/mnt/data/movies_tv_distribution.csv'

out_paths
