# Netflix Movies & TV Shows - Exploratory Data Analysis (EDA)

## Project Overview

This project was completed as part of the **Gray Interface '26 ML Team Induction Task**.

The main goal of this project was not to build any machine learning model. Instead, the focus was on cleaning the Netflix dataset, preparing it for analysis, and finding useful insights using Exploratory Data Analysis (EDA).

---

# Dataset

* **Dataset:** Netflix Movies and TV Shows
* **Source:** Kaggle
* **Environment:** Google Colab
* **Dataset Loading:** Programmatically using `kagglehub`

---

# Libraries Used

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* itertools
* collections
* re

---

# Data Engineering (How & Why)

## 1. Dataset Inspection

First, I checked the shape of the dataset, data types, missing values, and some sample rows. This helped me understand the structure of the data before cleaning it.

---

## 2. Handling Missing Values

The dataset contained missing values in columns like **date_added, rating, duration, country, cast, and director**.

### date_added

I removed extra spaces and converted this column into datetime format.

**Why?**

Because year, month, and weekday cannot be extracted from a normal string.

---

### rating

Missing values were filled using the most frequent rating (mode).

**Why?**

Only a small number of values were missing, so using the mode keeps the dataset consistent.

---

### duration

Missing values were also filled using the mode.

**Why?**

The number of missing values was very small compared to the whole dataset.

---

## 3. Feature Engineering

The original **duration** column contained two different types of information.

For Movies:

* Duration in minutes

For TV Shows:

* Number of seasons

Instead of keeping both together, I created two new columns:

* **movie_duration**
* **tv_seasons**

**Why?**

Keeping both values in one column makes numerical analysis difficult. Creating separate columns makes the data easier to analyze.

---

## 4. Unnesting Multi-value Columns

The following columns contained multiple values separated by commas:

* cast
* director
* country
* listed_in

These columns were split into lists and expanded using `explode()`.

**Why?**

Without exploding these columns, one cell containing multiple values would be treated as a single value. Exploding them allows accurate counting of actors, directors, countries, and genres.

---

## 5. Datetime Feature Extraction

After converting `date_added` into datetime format, I extracted:

* Year
* Month
* Day of Week

**Why?**

These new features help analyze when Netflix adds the most content.

---

# Visualizations & Insights

## 1. Movies vs TV Shows Over Time

**Insight:**

Before 2015, both Movies and TV Shows were added in relatively small numbers. After 2016, Netflix rapidly increased the number of titles added. Movies still dominate the platform, but TV Shows also grew very quickly over time.

---

## 2. Content Added by Month

**Insight:**

July had the highest number of content additions. This shows that Netflix adds a large amount of content during the middle of the year.

---

## 3. Content Added by Weekday

**Insight:**

Content is not added equally on every day of the week. Some weekdays have noticeably more releases than others.

---

## 4. Movie Duration Distribution

**Insight:**

Most Netflix movies have a duration close to 90–120 minutes, while very short and very long movies are less common.

---

## 5. Rating Distribution

**Insight:**

Only a few ratings make up most of the Netflix catalog, showing that the platform mainly focuses on a limited number of audience categories.

---

## 6. Top Content Producing Countries

**Insight:**

A small number of countries contribute most of the content available on Netflix, showing that production is concentrated in a few major countries.

---

## 7. Most Popular Genres

**Insight:**

Some genres appear much more frequently than others. This shows Netflix has a strong focus on a few popular content categories.

---

## 8. Country Collaboration Analysis

**Insight:**

Many Netflix titles are produced through collaborations between countries. The heatmap clearly shows which country pairs work together most often.

---

## 9. Actor–Director Analytics

**Insight:**

Some actors and directors have worked together multiple times, forming strong professional partnerships across several titles.

---

## 10. Genre Distribution of Top 10 Actors

**Insight:**

The most popular actors appear in many different genres instead of being limited to only one. Drama and International Movies were among the most common genres for several top actors.

---

## 11. Most Common Genre Combinations

**Insight:**

Netflix often assigns multiple genres to the same title. Some genre combinations appear much more frequently than others, showing common patterns in content classification.

---

## 12. Horror vs Comedy Text Analysis

**Insight:**

The descriptions of Horror titles mainly contain words related to fear, mystery, danger, and suspense. Comedy descriptions contain more words related to family, friendship, relationships, and fun. This shows a clear difference in the language used for both genres.

---

# Conclusion

This project helped me understand the complete process of Exploratory Data Analysis, starting from cleaning messy data to generating useful visualizations and insights.

During this project, I learned how to:

* Handle missing values
* Convert and clean datetime data
* Create new features
* Work with nested columns using `explode()`
* Build custom analysis using dictionaries and combinations
* Create different types of visualizations
* Draw meaningful insights from real-world data

Overall, this project improved my understanding of data preprocessing and EDA, which are important steps before building any machine learning model.
