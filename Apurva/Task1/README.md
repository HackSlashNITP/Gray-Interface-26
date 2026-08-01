# Netflix EDA

## About

In this project, I explored the Netflix Movies and TV Shows dataset. The main goal was to clean the data first and then use different visualizations to understand the dataset better.

The dataset was loaded directly from Kaggle using **kagglehub**, so no manual downloading was needed.

## Data Cleaning

* **Duration**

  * The `duration` column contained minutes for movies and seasons for TV shows.
  * I created two separate columns: `movie_duration` and `tv_seasons` so both could be analyzed properly.

* **Splitting Multiple Values**

  * Columns like **country**, **cast**, **director**, and **listed_in** had multiple values in a single cell.
  * I split them into separate rows so they could be counted and visualized correctly.

* **Date Conversion**

  * Converted `date_added` into datetime format.
  * Extracted the **year**, **month**, and **day of the week** for time-based analysis.

## Visualizations

* **Netflix Focus Shift**

  * Compared the number of Movies and TV Shows added over the last decade.
  * **Observation:** Movies are still more common, but TV Shows have grown steadily.

* **Monthly Content Additions**

  * Checked how content additions are distributed throughout the year.
  * **Observation:** Some months consistently have more releases than others.

* **Country Collaboration Heatmap**

  * Created a heatmap to see which countries frequently appear together in productions.
  * **Observation:** A few countries collaborate much more often than the rest.

* **Actor-Director Duos**

  * Found the actor-director pairs that have worked together the most.
  * **Observation:** Some collaborations appear repeatedly across multiple titles.

* **Genre Distribution of Top Actors**

  * Compared the genres of the actors with the highest number of appearances.
  * **Observation:** Most popular actors have worked across several genres.

* **Genre Combinations**

  * Found the genre pairs that occur together most frequently.
  * **Observation:** Certain combinations are much more common than others.

* **Horror vs Comedy Descriptions**

  * Compared the most common words used in descriptions of Horror and Comedy titles.
  * **Observation:** Horror descriptions mostly contain suspense-related words, while Comedy descriptions use lighter and humorous words.

## Conclusion

This project shows how important data cleaning is before doing any analysis. By separating durations, expanding multi-value columns, and processing dates, the dataset became much easier to work with. The visualizations helped reveal trends in Netflix's content, release patterns, collaborations between countries, popular actor-director pairs, genre distributions, and description patterns across different genres.
