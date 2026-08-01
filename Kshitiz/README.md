# 🎬 Netflix EDA — Gray Interface '26 | Task 1

**Dataset:** [Netflix Movies and TV Shows (Kaggle)](https://www.kaggle.com/datasets/shivamb/netflix-shows/data)  
**Environment:** Google Colab — data loaded programmatically via `kagglehub`  
**Libraries:** Pandas · NumPy · Matplotlib · Seaborn · WordCloud

---

## 1 · Data Engineering Pipeline

### Fixing the Duration Column

**The problem:** The `duration` column stores values like `"90 min"` for Movies and `"3 Seasons"` for TV Shows — two completely different units in one column.

**What I did:** Used `str.extract(r'(\d+)')` to pull out just the number. Then used `.where()` to split it into two separate columns — `movie_minutes` (only filled for Movies) and `tv_seasons` (only filled for TV Shows).

**Why:** You can't average or compare a movie's 90 minutes with a show's 3 seasons. Keeping them separate makes analysis correct.

---

### Parsing the Date Column

**The problem:** `date_added` is a raw string like `"January 15, 2021"` — useless for any time-based analysis.

**What I did:** Used `pd.to_datetime()` with `format='%B %d, %Y'` and `errors='coerce'` so bad values become NaT instead of crashing. Then extracted `year_added`, `month_added`, and `day_of_week`.

**Why:** Without parsing, we can't do any trend or seasonal analysis. The three extracted columns each answer a different question.

---

### Splitting Comma-Separated Columns

**The problem:** Columns like `listed_in`, `cast`, and `country` store multiple values as a single comma-separated string — e.g. `"Dramas, International Movies"`. Counting these directly gives wrong results.

**What I did:** Used `.str.split(',').explode().str.strip()` on each column. This gives one value per row so counts are accurate.

**Why:** Without exploding, a title tagged as both "Dramas" and "Comedies" would never be counted under either individually.

---

## 2 · Visualizations & Insights

| # | Plot | Insight |
|---|------|---------|
| 1 | **Movies vs TV Shows (Pie)** | Movies make up ~70% of the catalogue. Netflix started as a movie platform and still is one, though TV shows are growing. |
| 2 | **Titles Added Per Year (Stacked Bar)** | Content additions exploded from 2015–2019 and slowed after 2020. Netflix's growth phase was aggressive and has since stabilized. |
| 3 | **Titles Added by Month (Bar)** | January and December are the peak months. This matches holiday viewership spikes and new-year catalogue refreshes. |
| 4 | **Top 10 Genres (Bar)** | "International Movies" ranks #1 — showing Netflix's global strategy. "Dramas" and "Comedies" follow. |
| 5 | **Top 10 Countries (Bar)** | USA leads by a wide margin. India is a strong #2, a trend that only emerged after 2018 when Netflix invested heavily in Bollywood. |
| 6 | **Movie Duration Distribution (Histogram)** | Most movies cluster between 80–110 minutes with a median near 98 min. Very few exceed 150 min — these are rare epics. |
| 7 | **TV Show Seasons (Bar)** | Over 60% of TV shows have only 1 season — mostly limited series and international shows that don't run long. |
| 8 | **Content Ratings (Bar)** | TV-MA dominates, confirming Netflix targets adults. Children's content (TV-Y, G) exists but is a small fraction. |
| 9 | **Top 15 Actors (Bar)** | Anupam Kher leads — entirely driven by his large Bollywood filmography on Netflix. The top 20 is a mix of Indian, American, and Japanese actors. |
| 10 | **Top 10 Directors (Bar)** | The most prolific directors on Netflix tend to make documentaries and stand-up specials, not big feature films. |
| 11 | **Genre Co-occurrence Heatmap** | "Dramas" and "International Movies" almost always appear together. "Comedies" + "Romantic Movies" is another very common pair. Most titles are multi-genre tagged. |
| 12 | **Day of Week (Bar)** | Friday is the most common upload day by far — Netflix releases content right before the weekend when people have time to watch. |
| 13 | **WordClouds — Horror vs Comedy** | Horror descriptions use words like *dark, escape, must, young*. Comedy descriptions use *life, friends, love, family*. Both genres are about people — the difference is the stakes. |
| 14 | **Release Year Trend (Line)** | Movies on Netflix from the 1990s–2000s are older classics. Post-2010 content dominates, especially after 2015 when Netflix started producing original content. |

---

## 3 · Repo Structure

```
Kshitiz/
├── Netflix_EDA_GrayInterface26.ipynb   ← Full notebook
└── README.md                           ← This file
```

---

