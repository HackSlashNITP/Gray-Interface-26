Netflix Movies & TV Shows — EDA 

About this project

This is the submission for the Gray Interface '26 , Task 1. The goal: to extract, clean, and visualize as many meaningful insights as possible from the raw metadata. Used the Netflix Movies and TV Shows dataset from Kaggle.

Worked in Google Colab and pulled the dataset using kagglehub (as instructed), no manual CSV download.

python import Kagglehub
path = kagglehub.dataset_download("shivamb/netflix-shows")
df = pd.read_csv(f"{path}/netflix_titles.csv")


Part 1: Data Cleaning - How and Why

Before any plotting, the raw netflix_titles.csv needed some fixing. Here's what I did and the reason.

1. Filling missing values

Columns like director, cast, country, and duration had a lot of nulls. Instead of dropping those rows (which would've thrown away usable data for other columns), I filled them with 'unknown', and filled missing rating with 'Not rated'. This way a row missing a director can still be used when I'm analyzing genres or countries.

2. The duration fix

The duration column mixes two completely different things in one field: duration in minute for movies and duration in seasons for TV shows , can't be plotted or compare those directly. So:

splited the string on whitespace (duration.str.split()) to separate the number from the unit(minute and season).
Converted the numeric part to float.
Used '.where()' to create two separate columns for each type : movie_duration_min (only filled for movies) and TV_show_seasons (only filled for TV shows).
Added an assert check right after, to make sure every Movie row that has a value lines up correctly and nothing leaked across types. This caught  mistakes early instead of silently producing wrong charts later.


3. Unnesting the arrays

cast, director, country, and listed_in are comma-separated strings having multiple values into one cell (for example: "India, United States"). It's not possible to  count or group by these properly while they're still one long string — "India, United States" would just be treated as one unique value, separate from just "India".

To fix this, for each of those 4 columns:


Split on the comma into a Python list and stripped extra whitespace from each item.
Used df.explode() to turn each list into multiple rows — one row per value — while keeping the rest of the row's data the same.


This gave me cast_exploded, director_exploded, country_exploded, and genre_exploded — these became the base for almost every "top 10" and heatmap chart later.

4. Datetime parsing

date_added was a plain string with inconsistent leading whitespace in some rows. Stripped it, converted it with pd.to_datetime(), and pulled out year_added, month_added, and day_of_week_added as separate columns so I could group/plot by each of them independently.

5. Fixing a hidden data error in rating

While checking value_counts() on rating, I noticed values like '74 min', '84 min', '66 min' sitting in the ratings column — clearly a shifted/misaligned row , replaced those with 'Not rated' rather than guessing the original rating, since I had no reliable way to recover the correct value.


Part 2: Visualizations - What I Found

For every chart below: what it is and what I noticed when I looked at it.

1. Movie duration distribution (boxplot)

Shows the spread of movie runtimes. Most movies cluster around 87 - 114 minutes (the IQR),mean ~99.6 minutes, with a handful of outliers  -  some very short (under 30 min) and a few very long (3+ hours, max recorded is 312 minutes). The box plot made the outliers obvious in a way a single average number wouldn't.

2. Movies vs TV Shows count (countplot)

A simple count of type. Netflix's catalog is  roughly 70% Movies vs 30% TV Shows - movies make up the clear majority.

3. Movies vs TV Shows added per year (line chart)

This was one of the more interesting ones. Plotting count of movies and TV shows added by year_added shows both grew steadily until ~2018 - 2019 peak, then constant for TV Shows and decresing for movies and pulling back in 2020 - 2021 (the dataset itself only runs through September 2021, so the 2021 was undercounted). It also shows whether Netflix shifted its content strategy , movies still dominate in absolute count every year, but TV Shows close the gap somewhat in the later years.

4. Releases by month (countplot, grouped by type)

Shows which months get the most new content added and shows total releases , movie and TV shows.  December and July show the biggest spikes showing movies released more than tv shows.


5. Releases by day of week (countplot)

Same idea but for day of week , on which day most of release happen. Friday is the most common day which lines up with general industry practice of releasing big content right before the weekends.

6. Same-year release vs added (line chart)

This compares how many titles were added to Netflix the same year they were originally released, vs the total titles added that year. Same-year additions grow almost exponentially from 2014 onward, peaking at 777 titles in 2020 meaning Netflix is increasingly adding very current content rather than older catalog titles.

7. International co-production heatmap

Built by counting how often each pair of countries appears together in the same title's country field (loop through every title's country list, count every ordered pair). The heatmap of the top 10 countries shows 'United States - United Kingdom' is the strongest collaboration pair (279 co-productions), followed by 'United States - Canada' (217) and 'United States - France' (125) , which makes sense given the shared language and well-established film industries between them.  

8. Top 10 bar charts (Country / Actors / Genre / Directors)

Four simple horizontal bar charts (I wrote one reusable plot_top10() function for these instead of repeating the same plotting code four times). They show:


Top countries: United States(with 3690 titles), followed by India (1046) and the United Kingdom (806)
Top actors: Anupam Kher(with 43 titles), followed by Shah Rukh Khan (35) and Julie Tejwani (33)
Top genres: International Movies(with 2752 titles), followed by Dramas (2427) and Comedies (1674)
Top directors: Rajiv Chilaka(with 22 titles), followed by Jan Suter (21) and Raul Campos (19)


9. Top 10 Actor - Director duos

Counted using nested loops through every title's cast and director list, pairing each actor with each director on that title and tallying how often each pair repeats. The top duo - Julie Tejwani with Rajiv Chilaka and Rajesh Kava with Rajiv Chilaka, tied at 19 titles each, show a recurring collaboration - likely a director mostly works with the same lead actor across multiple projects.

10. Genre distribution heatmap - Top 10 Actors

Shows how each of the top 10 most-frequent actors' titles are spread across genres (raw counts, since each actor's total title count). Most of the top actors are Indian  (Anupam Kher, Shah Rukh Khan, Naseeruddin Shah, Om Puri, Akshay Kumar, Amitabh Bachchan) and the rest are Japanese actors (Takahiro Sakurai, Yuki Kaji) , the Indian actors concentrate in Dramas/International Movies, the Japanese actors concentrate in Anime Series/International TV Shows.

11. Genre distribution heatmap - Top 10 Directors

Same idea, for directors. Most directors concentrate almost all of their titles in a single genre column e.g. the comedy-special directors light up almost exclusively under "Stand-Up Comedy", while Rajiv Chilaka concentrates in "Children & Family Movies" / "Kids' TV" 

12. Genre distribution heatmap - Top 10 Countries (percentage-based)

This one needed a different approach than the actor/director version. Countries have wildly different Genre counts (the US alone has thousands), so a raw-count heatmap would just show the US lighting up every column and every other country looking empty by comparison just not readable much So instead:


I converted each country's row into a percentage of that country's own catalog (grid.div(grid.sum(axis=1), axis=0) * 100), so every country is compared on equal footing regardless of how many total titles it has.
I then dropped genre columns where no country reached even 5%, since the full grid had 40+ genre columns and most were near-zero for every country - keeping them just made the heatmap unreadable.


What this revealed:  South Korea shows a strong skew toward TV Dramas — a much higher percentage of its catalog than other countries - likely reflecting the K-drama boom. India shows a notable skew toward Dramas/International Movies. The US, having the largest and most diverse catalog, shows a flatter spread across most of genres. Japan mostly focused on Anime. Most of the top 10 country focus on International movies , Drama.
Exact per-cell percentages are best read directly off the rendered heatmap image in the notebook.

The general pattern: smaller-catalog countries tend to specialize in fewer genres, while large-catalog countries spread more evenly - simply because they have enough volume to cover everything.

13. Most common genre combinations

Counted how often exact combinations of genres appear together on the same title (sorted and joined as one key per title, then counted). "Dramas & International Movies" is the most common combo(with 362 titles), followed closely by "Documentaries" (359) and "Stand-Up Comedy" (334) - showing that Netflix rarely tags titles with just one genre; cross-genre tagging (Dramas + International Movies, Comedies + Dramas + International Movies, etc.) is the norm rather than the exception.

14. Most frequent words in descriptions: Horror vs Comedy

Wrote a function that filters titles by genre, joins all their descriptions into one block of text, and counts word frequency (excluding short/common filler words). Comparing Horror and Comedy side by side:


Horror descriptions lean toward words like : mysterious (32), horror (28), supernatural (25), begins (21), spirit (18), terrifying (18), haunted (18)
Comedy descriptions lean toward words like  : comedy (124), comedian (116), three (112), takes (111), finds (92), school (90), comic (90), special (68), world (67)


This shows the kind of language/storytelling each genre tends to use even at the description level — Horror tends toward isolation/danger-type words
Comedy tends toward performer/social-situation words (comedian, comic, school)  .


Tools used

pandas, numpy, matplotlib, seaborn 


What I'd add with more time


A proper network graph (using networkx) for the country co-production data, as an alternative view to the heatmap.
A word cloud for genre descriptions.



Notebook

The full notebook with all code and outputs is in Task_1_Gray_interface.ipynb in this repo.