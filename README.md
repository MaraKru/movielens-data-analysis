# MovieLens Analytics: Group Data Analysis Project

## About the Project

Analytical report based on the MovieLens dataset (small version) using Python and Jupyter Notebook. The goal of our project was not only to perform data analysis but also to create modular, testable code architecture, applying skills in web scraping, data processing, mathematical analysis, and report generation.

### My Role: Team Lead

I was responsible for:
- developing the Ratings class (processing and analyzing movie rating data)
- writing tests for the Ratings class
- code refactoring and optimization (fixing problematic code sections, particularly in the Links class, which accelerated cache parsing and data loading)
- team coordination (task distribution, organizing meetings, timeline management)
- creating the final report for the Ratings class + overall report editing


### Technologies & Tools

* Language: Python 3
* Libraries: pandas, pytest, requests, BeautifulSoup, collections, datetime, re, json
* Tools: Jupyter Notebook, Git, IMDb scraping


### Quick Start

```bash
python -m pytest .\movielens_analysis.py -v   # run tests

# view results of each module in movielens_report.ipynb
```


### What We Implemented

*Module movielens_analysis.py*

Contains 5 main classes:
- Ratings (my responsibility): Analysis of rating distribution over time, metric calculations (mean, median, variance), user segmentation
- Movies: film analysis, distribution by release year and genres
- Tags: tag processing - most popular, longest, word search
- Links: IMDb data scraping (budget, director, box office) with caching
- Tests: writing tests for each method (data type verification, sorting correctness, calculation accuracy), 50+ tests total


*Testing*

- Tests for every method
- Data type verification, sorting correctness checking
- Run: `python -m pytest movielens_analysis.py -v`


*Analytical Report*

- Utilization of all module methods
- Data-driven narrative
- Each cell includes %timeit for performance evaluation


### What I Learned

**HARD SKILLS**
* Python (OOP, data structures, algorithms)
* Data Analysis (pandas, statistics)
* Web scraping from external sources (BeautifulSoup)
* Testing


**SOFT SKILLS**
* Project management
* Team collaboration and leadership (3-people team)
* Results presentation
* Time management :)
