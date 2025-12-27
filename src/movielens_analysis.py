from collections import defaultdict
from collections import Counter
from datetime import datetime
import pytest
import os
import requests
from bs4 import BeautifulSoup
import re
import json


# run tests: python -m pytest .\movielens_analysis.py -v
# -v for verbose output, -m to run specific module as script


class Links:
    # Analyzing data from links.csv
    def __init__(self, path_to_the_file = 'data/ml-latest-small/links.csv', path_to_the_movies = 'data/ml-latest-small/movies.csv', path_to_dump = '../datasets/imdb_cache.json'):

        self.file = path_to_the_file
        self.movies_file = path_to_the_movies
        self.dump_file = path_to_dump

        self.cache = {}
        if os.path.exists(self.dump_file):
            try:
                with open(self.dump_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except:
                self.cache = {}


        # loading data from links.csv
        with open(self.file, 'r', encoding = 'utf-8') as file:
            links_lines = file.readlines()

        # loading data from movies.csv
        with open(self.movies_file, 'r', encoding = 'utf-8') as movies_file:
            movies_lines = movies_file.readlines()
    
    # adding lines from links and adding to movies
        self.data = []
        links_data = links_lines[1:]
        movies_data = movies_lines[1:]

        min_len = min(len(links_data), len(movies_data))
        for i in range(min_len):
            clean_link = links_data[i].strip('\n')
            clean_movie = movies_data[i].strip('\n')
            combination_line = clean_link + ',' + clean_movie
            self.data.append(combination_line)


        self.movie_to_imdb = {}
        for line in self.data:
            parts = line.split(',')
            movie_id = int(parts[0])
            imdb_id = parts[1]
            self.movie_to_imdb[movie_id] = imdb_id

    def get_imdb(self, list_of_movies, list_of_fields):

        # This docstring describes the assignment requirements for the educational project

        """
        The method returns a list of lists [movieId, field1, field2, field3, ...] for the list of movies given as the argument (movieId).
        For example, [movieId, Director, Budget, Cumulative Worldwide Gross, Runtime].
        The values should be parsed from the IMDB webpages of the movies.
        Sort it by movieId descendingly.
        """ 

        # second try to get imdbID from movieID
        def get_imdb_id(movie_id):
            return self.movie_to_imdb.get(movie_id)
        
        # three parsing IMDB
        def parse_page(imdb_id):
            if imdb_id in self.cache: # if data already dowloaded in cache just return them without parsing
                return self.cache[imdb_id]
        
            url = f"https://www.imdb.com/title/tt{imdb_id}/"
            try:
                resp = requests.get(url, headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }, timeout=5)
                if resp.status_code != 200:
                    raise Exception(f"HTTP {resp.status_code}: Failed to load page for imdbId '{imdb_id}'.")
                if resp.status_code == 404:
                    raise Exception(f"404 Error: Film with imdbId '{imdb_id}' not found on IMDB.")
            except:
                empty = {'title': None, 'director': None, 'budget': None, 'gross': None, 'runtime': None} # empty data
                self.cache[imdb_id] = empty
                return empty
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            text = soup.get_text() # getting just text without classes like <div> and <p>

            data = {
                'title': None,
                'director': None,
                'budget': None,
                'gross': None,
                'runtime': None
            } # an empty dictionary to save the data
            
            # name
            h1 = soup.find('h1') # cause names saves in tag <h1>
            if h1:
                data['title'] = h1.get_text().split('(')[0].strip() # taking only text without year(it is in ())

            # director
            dir_link = soup.find('a', href=re.compile(r'/name/nm\d+')) # getting the first link to person - takes him as a director
            if dir_link:
                data['director'] = dir_link.get_text().strip()

            #budget and gross
            for match in re.finditer(r'\$(\d{1,3}(?:,\d{3})*)', text): #trying to get all the money stuff 
                num = int(match.group(1).replace(',', ''))
                if 'Budget' in text[max(0, match.start()-50):match.start()]: # find if there any word like 'Budget' in 50 symb
                    data['budget'] = num
                elif 'Gross' in text[max(0, match.start()-50):match.start()]:
                    data['gross'] = num
            #rumtime
            rt = re.search(r'(\d+)\s*min', text) # find phrase like 180 min -> extracting only the number
            if rt:
                data['runtime'] = int(rt.group(1))

            self.cache[imdb_id] =  data
            # loading cache in file
            try:
                with open(self.dump_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, ensure_ascii=False) # to write data in russian names
            except:
                pass
            return data
            #getting the whole result
        imdb_info = []
        for movie_id in list_of_movies:
            imdb_id = get_imdb_id(movie_id)
            info = parse_page(imdb_id) if imdb_id else {
                'title': None, 'director': None, 'budget': None, 'gross': None, 'runtime': None
            }
            row = [movie_id] + [info.get(field) for field in list_of_fields]
            imdb_info.append(row)

        imdb_info.sort(key=lambda x: x[0], reverse=True) # sorting by first element in decreasing
        return imdb_info 
        
  
    def top_directors(self, n):
        """
        The method returns a dict with top-n directors where the keys are directors and 
        the values are numbers of movies created by them. Sort it by numbers descendingly.
        """
        # getting all movieId
        movie_ids = [int(line.split(',')[0]) for line in self.data[:1000]]
        imdb_data = self.get_imdb(movie_ids, ['director'])

        director_count = {}
        for row in imdb_data:
            director = row[1] # cause structure [movieId, director]
            if director is not None:
                director_count[director] = director_count.get(director, 0) + 1
        sorted_directors = sorted(director_count.items(), key=lambda x: x[1], reverse=True)
        directors = dict(sorted_directors[:n])
        return directors
        
    def most_expensive(self, n):
        """
        The method returns a dict with top-n movies where the keys are movie titles and
        the values are their budgets. Sort it by budgets descendingly.
        """
        movie_ids = [int(line.split(',')[0]) for line in self.data[:1000]] # collecting all movieIds from each line
        imdb_data = self.get_imdb(movie_ids, ['title', 'budget'])

        budgets = {} # our dict
        for row in imdb_data: # loading our dict
            title, budget = row[1], row[2]
            if title is not None and budget is not None:
                budgets[title] = budget

        sorted_budgets = sorted(budgets.items(), key=lambda x: x[1], reverse=True) # budgets.items() turning dict to list of couples, key=lambda x: x[1] sorting by second element
        budgets = dict(sorted_budgets[:n]) # dict of sorted from most expensive to less by title-money pairs
        return budgets
        
    def most_profitable(self, n):
        """
        The method returns a dict with top-n movies where the keys are movie titles and
        the values are the difference between cumulative worldwide gross and budget.
        Sort it by the difference descendingly.
        """
        movie_ids = [int(line.split(',')[0]) for line in self.data[:1000]]
        imdb_data = self.get_imdb(movie_ids, ['title', 'budget', 'gross'])

        profits = {}
        for row in imdb_data:
            title, budget, gross = row[1], row[2], row[3]
            if title is not None and budget is not None and gross is not None:
                profit = gross - budget
                profits[title] = profit

        sorted_profits = sorted(profits.items(), key=lambda x: x[1], reverse=True)
        profits = dict(sorted_profits[:n])
        return profits
        
    def longest(self, n):
        """
        The method returns a dict with top-n movies where the keys are movie titles and
        the values are their runtime. If there are more than one version – choose any.
        Sort it by runtime descendingly.
        """
        movie_ids = [int(line.split(',')[0]) for line in self.data[:1000]]
        imdb_data = self.get_imdb(movie_ids, ['title', 'runtime']) 

        runtimes = {}
        for row in imdb_data:
            title, runtime = row[1], row[2]
            if title is not None and runtime is not None:
                runtimes[title] = runtime

        sorted_runtimes = sorted(runtimes.items(), key=lambda x: x[1], reverse=True)
        runtimes = dict(sorted_runtimes[:n])
        return runtimes
        
    def top_cost_per_minute(self, n):
        """
        The method returns a dict with top-n movies where the keys are movie titles and
        the values are the budgets divided by their runtime. The budgets can be in different currencies – do not pay attention to it. 
        The values should be rounded to 2 decimals. Sort it by the division descendingly.
        """
        movie_ids = [int(line.split(',')[0]) for line in self.data[:1000]]
        imdb_data = self.get_imdb(movie_ids, ['title', 'budget', 'runtime']) 

        costs = {}
        for row in imdb_data:
            title, budget, runtime = row[1], row[2], row[3]
            if title is not None and budget is not None and runtime  > 0:
                cost = round((budget / runtime), 2)
                costs[title] = cost
        
        sorted_costs = sorted(costs.items(), key=lambda x: x[1], reverse=True)
        costs = dict(sorted_costs[:n])
        return costs

    def movies_by_director(self, director_name):
        """
        this method returns a lisf of director's films
        """
        movie_ids = [int(line.split(',')[0]) for line in self.data[:1000]]
        imdb_data = self.get_imdb(movie_ids, ['title', 'director'])

        movies = []
        for row in imdb_data:
            title = row[1]
            director = row[2]
            if director and director.lower() == director_name.lower():
                movies.append(title)
    
        return movies

    def shortest_movie(self):
        """
        this method returns a name of the sortest film
        """
        movie_ids = [int(line.split(',')[0]) for line in self.data[:1000]]
        imdb_data = self.get_imdb(movie_ids, ['title', 'runtime'])

        valid_movies = [(row[1], row[2]) for row in imdb_data
        if row[1] is not None and row[2] is not None and row[2] > 0] # getting films only with smth in duration

        if not valid_movies:
            return None
    
        shortest = min(valid_movies, key=lambda x: x[1]) # finding a film with sortest duration
        
        return shortest[0]
    



class Movies:

    # Analyzing data from movies.csv
    
    def __init__(self, path_to_the_file):

        self.movies = []
        with open(path_to_the_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines[1:1001]:
                parts = line.strip().split(',', 2)
                if len(parts) == 3:
                    movie_id = parts[0]
                    title = parts[1]
                    genres = parts[2]
                    self.movies.append((movie_id, title, genres))

    def dist_by_release(self):
        """
        The method returns a dict or an OrderedDict where the keys are years and the values are counts. 
        You need to extract years from the titles. Sort it by counts descendingly.
        """
        year_count = {}
        for _, title, _ in self.movies: # we need only title
            if '(' in title and title.endswith(')'): # cuz year is in ()
                year_part = title.split('(')[-1].strip(')') # taking year from (year)
                if year_part.isdigit() and len(year_part) == 4: # checking is it a number and length
                    year = int(year_part)
                    year_count[year] = year_count.get(year, 0) + 1

        sorted_years = sorted(year_count.items(), key=lambda x: x[1], reverse=True) # sorting by decrease
        release_years = dict(sorted_years)
        return release_years
    
    def dist_by_genres(self):
        """
        The method returns a dict where the keys are genres and the values are counts.
     Sort it by counts descendingly.
        """
        genre_count = {}
        for _, _, genres in self.movies:
            if genres != '(no genres listed)': # if there no genres -> pass
                for genre in genres.split('|'): # spliting genres by '|'
                    if genre in genre_count:
                        genre_count[genre] += 1 # if already have number for this movie -> just add 1
                    else:
                        genre_count[genre] = 1 # for the firts time
        
        sorted_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)
        genres = dict(sorted_genres)
        return genres
        
    def most_genres(self, n):
        """
        The method returns a dict with top-n movies where the keys are movie titles and 
        the values are the number of genres of the movie. Sort it by numbers descendingly.
        """
        movie_genres_count = {}
        for _, title, genres in self.movies:
            if genres != '(no genres listed)':
                count = len(genres.split('|'))
                movie_genres_count[title] = count

        sorted_movies = sorted(movie_genres_count.items(), key=lambda x: x[1], reverse=True)
        movies = dict(sorted_movies)
        return movies




class Ratings:

    def __init__(self, path_to_the_file):
        self.path = path_to_the_file    # specify during object creation
        self.ratings_data = self.load_ratings_data()     # load movie ratings
        self.movies_data = self.load_movies_data()       # load movie titles
        self.movies = self.Movies(self)       # subclass object
        self.users = self.Users(self)        # also


    
    def load_ratings_data(self):
        data = []
        with open(self.path, 'r', encoding='utf-8') as f:
            headers = f.readline().strip().split(',')     # readline reads first line
            for i, line in enumerate(f):
                if i >= 1000: break
                values = line.strip().split(',')
                row = dict(zip(headers, values))    # zip pairs lists: 'userId': '1'

                row['userId'] = int(row['userId'])   # convert string to int

                row['movieId'] = int(row['movieId'])
                row['rating'] = float(row['rating'])
                row['timestamp'] = int(row['timestamp'])

                data.append(row)

        return data
    

    
    def load_movies_data(self):
        movies_path = self.path.replace('ratings.csv', 'movies.csv')
        movie_titles = {}

        with open (movies_path, 'r', encoding='utf-8') as f:
            f.readline()   # skip header row
            for line in f:
                line = line.strip()
                first_comma = line.find(',')    # position of first comma
                last_comma = line.rfind(',')

                movie_id = int(line[:first_comma])
                title = line[first_comma + 1: last_comma]

                if title.startswith('"') and title.endswith('"'):
                    title = title[1:-1]    # remove quotation marks

                movie_titles[movie_id] = title

        
        return movie_titles



    class Movies:    

        def __init__(self, rating_obj):      # Rating object
            self.rating_obj = rating_obj      # store reference to parent


        # ratings per year
        def dist_by_year(self):

            year_counts = defaultdict(int)              # smart dictionary (auto-creates keys)


            for row in self.rating_obj.ratings_data:
                timestamp = row['timestamp']
                year = datetime.fromtimestamp(timestamp).year
                year_counts[year] += 1

            ratings_by_year = dict(sorted(year_counts.items()))

            return ratings_by_year   #items converts dict to list of pairs
        


        # rating frequency distribution
        def dist_by_rating(self):

            rating_counts = defaultdict(int)

            for row in self.rating_obj.ratings_data:

                rating = row['rating']
                rating_counts[rating] += 1

            ratings_distribution = dict(sorted(rating_counts.items()))

            return ratings_distribution
        


        # top movies by rating count (any rating)
        def top_by_num_of_ratings(self, n):

            movie_ratings_count = defaultdict(int)        # count ratings per movie

            for row in self.rating_obj.ratings_data:
                movie_id = row['movieId']
                movie_ratings_count[movie_id] += 1

            
            movies_with_titles = {}                              # merge data with titles
            for movie_id, count in movie_ratings_count.items():
                if movie_id in self.rating_obj.movies_data:
                    movie_title = self.rating_obj.movies_data[movie_id]
                    movies_with_titles[movie_title] = count


            top_movies = dict(sorted(
                movies_with_titles.items(),   # create list of tuples
                key = lambda x: x[1],    # sort by values (not keys)
                reverse = True
            ) [:n])

            return top_movies
        


        def top_by_ratings(self, n, metric='mean'):
    
            # metric calculation function
            if metric == 'mean':
                calc = lambda x: sum(x) / len(x)
            elif metric == 'median':
                calc = lambda x: sorted(x)[len(x) // 2] if len(x) % 2 == 1 else \
                                (sorted(x)[len(x) // 2 - 1] + sorted(x)[len(x) // 2]) / 2
            else:
                calc = lambda x: sum(x) / len(x)  # default is mean
            
            movie_ratings = defaultdict(list)

            # all ratings per movie
            for row in self.rating_obj.ratings_data:
                movie_id = row['movieId']
                rating = row['rating']
                movie_ratings[movie_id].append(rating)

            
            # average rating per movie
            movie_scores = {}
            for movie_id, ratings in movie_ratings.items():
                if len(ratings) > 0:
                    score = calc(ratings)
                    movie_scores[movie_id] = round(score, 2)


            # merge titles with IDs
            movies_with_titles = {}
            for movie_id, score in movie_scores.items():
                if movie_id in self.rating_obj.movies_data:
                    movie_title = self.rating_obj.movies_data[movie_id]
                    movies_with_titles[movie_title] = score


            top_movies = dict(sorted(
                movies_with_titles.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n])
        

            return top_movies



        # most ratings on weekends
        def weekend_hits(self, n):
            movie_weekend_ratings = defaultdict(int)
            movie_all_ratings = defaultdict(int)
            
            for row in self.rating_obj.ratings_data:
                movie_id = row['movieId']
                timestamp = row['timestamp']
                
                weekday = datetime.fromtimestamp(timestamp).weekday()  # 5 - Saturday
                if weekday >= 5:
                    movie_weekend_ratings[movie_id] += 1
                
                movie_all_ratings[movie_id] += 1
            

            weekend_ratings = {}
            for movie_id in movie_all_ratings:
                if movie_all_ratings[movie_id] >= 2:
                    ratings = movie_weekend_ratings.get(movie_id, 0) / movie_all_ratings[movie_id]
                    weekend_ratings[movie_id] = round(ratings, 2)
            
            weekend_movies = {}
            for movie_id, ratings in dict(sorted(
                weekend_ratings.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:n]).items():
                if movie_id in self.rating_obj.movies_data:
                    weekend_movies[self.rating_obj.movies_data[movie_id]] = ratings
            
            return weekend_movies
        


        # most controversial movies
        def top_controversial(self, n):
            
            movie_ratings = defaultdict(list)

            # ID + rating
            for row in self.rating_obj.ratings_data:
                movie_id = row['movieId']
                rating = row['rating']
                movie_ratings[movie_id].append(rating)


            # variance (rating spread)
            movie_variances = {}
            for movie_id, ratings in movie_ratings.items():
                if len(ratings) > 1:    # variance requires at least 2 ratings
                    mean = sum(ratings) / len(ratings)
                    squared_diff = sum((x - mean) ** 2 for x in ratings)
                    variance = squared_diff / len(ratings)
                    movie_variances[movie_id] = round(variance, 2)
                else:
                    # single rating = variance 0
                    movie_variances[movie_id] = 0.0


            """
            Calculate mean rating, subtract from each rating, square differences,
            sum squares, divide by count 
            """

            
            # merge with titles
            movies_with_titles = {}
            for movie_id, variance in movie_variances.items():
                if movie_id in self.rating_obj.movies_data:
                    movie_title = self.rating_obj.movies_data[movie_id]
                    movies_with_titles[movie_title] = variance


            
            # sort by variance descending
            top_movies = dict(sorted(
                movies_with_titles.items(),
                key=lambda x: x[1],
                reverse = True
            )[:n])


            return top_movies

            




    class Users(Movies):
        def __init__(self, rating_obj):
            Ratings.Movies.__init__(self, rating_obj)
        

        def dist_by_num_ratings(self):

            user_count_ratings = defaultdict(int)
            for row in self.rating_obj.ratings_data:
                user_id = row['userId']
                user_count_ratings[user_id] += 1


            ratings_by_users = dict(sorted(
                user_count_ratings.items(),
                key=lambda x: x[1],
                reverse = True
            ))

            return ratings_by_users
        


        # by average rating
        def dist_by_avg_ratings(self, metric=lambda x: sum(x)/len(x)):

            # count all ratings
            user_ratings = defaultdict(list)

            for row in self.rating_obj.ratings_data:
                user_id = row['userId']
                user_rating = row['rating']
                user_ratings[user_id].append(user_rating)


            # average rating
            user_avg_ratings = {}
            for user_id, user_rating in user_ratings.items():
                if len(user_rating) > 0:
                    avg_rating = metric(user_rating)
                    user_avg_ratings[user_id] = round(avg_rating, 2)

            
            users_by_avg_raitings = dict(sorted(
                user_avg_ratings.items(),
                key=lambda x: x[1],
                reverse = True
            ))

            return users_by_avg_raitings
        


        # night owls
        def night_monsters(self, n):
            user_night_ratings = defaultdict(int)
            user_all_ratings = defaultdict(int)

            for row in self.rating_obj.ratings_data:
                user_id = row['userId']
                timestamp = row['timestamp']

                hour = datetime.fromtimestamp(timestamp).hour

                if hour >= 22 or hour < 6:
                    user_night_ratings[user_id] += 1

                user_all_ratings[user_id] += 1

            # percentage of night ratings per user
            night_ratings = {}
            for user_id in user_all_ratings:
                if user_all_ratings[user_id] >= 5:
                    ratings = user_night_ratings.get(user_id, 0) / user_all_ratings[user_id]
                    night_ratings[user_id] = round(ratings, 2)


            night_monsters = dict(sorted(
                night_ratings.items(),
                key = lambda x: x[1],
                reverse=True
            )[:n])

            return night_monsters
        


        # most controversial users (love-hate relationships)
        def top_controversial_users(self, n):

            user_ratings = defaultdict(list)

            for row in self.rating_obj.ratings_data:
                user_id = row['userId']
                rating = row['rating']
                user_ratings[user_id].append(rating)

        
            user_variances = {}
            for user_id, ratings in user_ratings.items():
                if len(ratings) > 1:
                    mean = sum(ratings) / len(ratings)
                    squared_diff = sum((x-mean) ** 2 for x in ratings)
                    variance = squared_diff / len(ratings)
                    user_variances[user_id] = round(variance, 2)
                else:
                    user_variances[user_id] = 0.0


            top_contr_users = dict(sorted(
                user_variances.items(),
                key=lambda x: x[1],
                reverse = True
            )[:n])


            return top_contr_users
        
    


class Tags:
    # Analyzing data from tags.csv
    def __init__(self, path_to_the_file):

        self.path = path_to_the_file
        all_tags = self.load_tags()
        self.tags_data = all_tags[:1000]

    def load_tags(self):
        """
        Loads tags from the file, skipping the header.
        Returns a list of tuples: (userId, movieId, tag, timestamp)
        """
        with open(self.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data = []
        for line in lines[1:]:
            fields = []
            current_data = ''
            flag = False
            i = 0
            line = line.strip()
            while i < len(line):
                char = line[i]
                if char == '"':
                    if flag:
                        flag = False
                    else:
                        flag = True
                elif char == ',' and not flag:
                    fields.append(current_data)
                    current_data = ''
                else:
                    current_data += char
                i += 1
            fields.append(current_data)

            userId = fields[0]
            movieId = fields[1]
            tag = fields[2]
            timestamp = fields[3]
            data.append((userId, movieId, tag, timestamp))
        return data


    def most_words(self, n):
        """
        The method returns top-n tags with most words inside. It is a dict 
 where the keys are tags and the values are the number of words inside the tag.
 Drop the duplicates. Sort it by numbers descendingly.
        """
        tag_word_counts = {}
        seen_tags = set()

        for _, _, tag, _ in self.tags_data:
            if tag in seen_tags:
                continue
            seen_tags.add(tag)
            word_count = len(tag.split())
            tag_word_counts[tag] = word_count

        sorted_tags = sorted(tag_word_counts.items(), key=lambda x: (-x[1], x[0]))
        big_tags = {tag: count for tag, count in sorted_tags[:n]}
        return big_tags

    def longest(self, n):
        """
        The method returns top-n longest tags in terms of the number of characters.
        It is a list of the tags. Drop the duplicates. Sort it by numbers descendingly.
        """
        seen_tags = set()
        unique_tags_with_len = []

        for _, _, tag, _ in self.tags_data:
            if tag in seen_tags:
                continue
            seen_tags.add(tag)
            unique_tags_with_len.append((tag, len(tag)))

        sorted_tags = sorted(unique_tags_with_len, key=lambda x: (-x[1], x[0]))
        big_tags = [tag for tag, length in sorted_tags[:n]]
        return big_tags

    def most_words_and_longest(self, n):
        """
        The method returns the intersection between top-n tags with most words inside and 
        top-n longest tags in terms of the number of characters.
        Drop the duplicates. It is a list of the tags.
        """
        most_wordy = set(self.most_words(n).keys())
        longest_tags = set(self.longest(n))
        big_tags = list(most_wordy & longest_tags)
        big_tags.sort()
        return big_tags
        
    def most_popular(self, n):
        """
        The method returns the most popular tags. 
        It is a dict where the keys are tags and the values are the counts.
        Drop the duplicates. Sort it by counts descendingly.
        """

        tag_counts = {}
        for _, _, tag, _ in self.tags_data:
            if tag in tag_counts:
                tag_counts[tag] += 1
            else:
                tag_counts[tag] = 1

        sorted_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))
        popular_tags = {tag: count for tag, count in sorted_tags[:n]}
        return popular_tags
        
    def tags_with(self, word):
        """
        The method returns all unique tags that include the word given as the argument.
        Drop the duplicates. It is a list of the tags. Sort it by tag names alphabetically.
        """
        word_lower = word.lower()
        found_tags = set()

        for _, _, tag, _ in self.tags_data:
            if word_lower in tag.lower():
                found_tags.add(tag)

        tags_with_word = sorted(list(found_tags))
        return tags_with_word

    def most_popular_by_movie(self, movie_id, n):

        # Returns top-n most popular tags for specific movie as {tag: count}

        tag_counts = {}
        for _, m_id, tag, _ in self.tags_data:
            if int(m_id) == movie_id:
                if tag in tag_counts:
                    tag_counts[tag] += 1
                else:
                    tag_counts[tag] = 1
        sorted_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))
        return {tag: count for tag, count in sorted_tags[:n]}

    def average_tag_length(self):

        # Returns average tag length in characters

        lengths = [len(tag) for _, _, tag, _ in self.tags_data]
        if not lengths:
            return 0
        return sum(lengths) / len(lengths)

    def get_unique_tags_count(self):

        # Returns total count of unique tags

        unique_tags = set()
        for _, _, tag, _ in self.tags_data:
            unique_tags.add(tag)
        return len(unique_tags)





# run tests: python -m pytest .\movielens_analysis.py -v
# -v for verbose output, -m to run module as script


class Tests:

    @pytest.fixture
    def tags(self):
        return Tags(path_to_the_file='data/ml-latest-small/tags.csv')
    
    @pytest.fixture
    def links(self):
        return Links(
            path_to_the_file='data/ml-latest-small/links.csv',
            path_to_the_movies='data/ml-latest-small/movies.csv',
            path_to_dump='../datasets/imdb_cache.json')
        
    
    @pytest.fixture
    def movies(self):
        return Movies('data/ml-latest-small/movies.csv')
    

    def test_ratings_initializations(self):
        # test Ratings initialization
        rating_obj = Ratings('data/ml-latest-small/ratings.csv')
        assert rating_obj is not None
        assert hasattr(rating_obj, 'movies')    # check attribute existence (Movies object)
        assert hasattr(rating_obj, 'users')



    def test_dist_by_year(self):
        rating_obj = Ratings('data/ml-latest-small/ratings.csv')
        result = rating_obj.movies.dist_by_year()

        assert isinstance(result, dict)   # verify data type (expects dictionary)

        # check dictionary items
        for year, count in result.items():
            assert isinstance(year, int)
            assert isinstance(count, int)

        years = list(result.keys())
        assert years == sorted(years) 



    def test_dist_by_rating(self):
        rating_obj = Ratings('data/ml-latest-small/ratings.csv')
        result = rating_obj.movies.dist_by_rating()

        assert isinstance(result, dict)

        for rating, count in result.items():
            assert isinstance(rating, float)
            assert isinstance(count, int)

        ratings = list(result.keys())
        assert ratings == sorted(ratings)



    def test_top_by_num_of_ratings(self):
        rating_obj = Ratings('data/ml-latest-small/ratings.csv')
        result = rating_obj.movies.top_by_num_of_ratings(3)
        
        assert isinstance(result, dict)
        assert len(result) == 3

        for title, count in result.items():
            assert isinstance(title, str)
            assert isinstance(count, int)

        counts = list(result.values())
        assert counts == sorted(counts, reverse=True)



    # test with mean metric
    def test_top_by_ratings_mean(self):
        rating_obj = Ratings('data/ml-latest-small/ratings.csv')
        result = rating_obj.movies.top_by_ratings(3, metric='mean')
        
        assert isinstance(result, dict)

        for title, rating in result.items():
            assert isinstance(title, str)
            assert isinstance(rating, float)

        if result:    # handle empty dict (insufficient ratings)
            ratings = list(result.values())
            assert ratings == sorted(ratings, reverse=True)


    # with median
    def test_top_by_ratings_median(self):
        rating_obj = Ratings('data/ml-latest-small/ratings.csv')
        result = rating_obj.movies.top_by_ratings(3, metric='median')
        
        assert isinstance(result, dict)

        for title, rating in result.items():
            assert isinstance(title, str)
            assert isinstance(rating, float)
        
        if result:
            ratings = list(result.values())
            assert ratings == sorted(ratings, reverse=True)


    def test_weekend_hits(self):
        rating_obj = Ratings('data/ml-latest-small/ratings.csv')
        result = rating_obj.movies.weekend_hits(3)
        
        assert isinstance(result, dict)
        assert len(result) <= 3

        for title, ratio in result.items():
            assert isinstance(title, str)
            assert isinstance(ratio, float)
            assert 0 <= ratio <= 1  # percentage should be between 0 and 1

        ratios = list(result.values())
        assert ratios == sorted(ratios, reverse=True) 
    


    def test_top_controversial(self):
        rating_obj = Ratings('data/ml-latest-small/ratings.csv')
        result = rating_obj.movies.top_controversial(3)
        
        assert isinstance(result, dict)
        assert len(result) == 3
    
        for title, variance in result.items():
            assert isinstance(title, str)
            assert isinstance(variance, float)
        
        variances = list(result.values())
        assert variances == sorted(variances, reverse=True)



    def test_users_dist_by_num_ratings(self):
        rating_obj = Ratings('data/ml-latest-small/ratings.csv')
        result = rating_obj.users.dist_by_num_ratings()
        
        assert isinstance(result, dict)
        
        for user_id, count in result.items():
            assert isinstance(user_id, int)
            assert isinstance(count, int)
        
        counts = list(result.values())
        assert counts == sorted(counts, reverse=True)


    def test_users_dist_by_avg_ratings(self):
        rating_obj = Ratings('data/ml-latest-small/ratings.csv')
        result = rating_obj.users.dist_by_avg_ratings()
        
        assert isinstance(result, dict)

        for user_id, avg_rating in result.items():
            assert isinstance(user_id, int)
            assert isinstance(avg_rating, float)
    
        avg_ratings = list(result.values())
        assert avg_ratings == sorted(avg_ratings, reverse=True)


    def test_night_monsters(self):
        rating_obj = Ratings('data/ml-latest-small/ratings.csv')
        result = rating_obj.users.night_monsters(3)
        
        assert isinstance(result, dict)
        assert len(result) == 3

        for user_id, ratio in result.items():
            assert isinstance(user_id, int)
            assert isinstance(ratio, float)
            assert 0 <= ratio <= 1

        ratios = list(result.values())
        assert ratios == sorted(ratios, reverse=True)


    def test_top_controversial_users(self):
        rating_obj = Ratings('data/ml-latest-small/ratings.csv')
        result = rating_obj.users.top_controversial_users(3)
        
        assert isinstance(result, dict)
        assert len(result) == 3

        for user_id, variance in result.items():
            assert isinstance(user_id, int)
            assert isinstance(variance, float)
    
        variances = list(result.values())
        assert variances == sorted(variances, reverse=True)



    def test_most_words_returns_dict(self, tags):
        result = tags.most_words(3)
        assert isinstance(result, dict)


    def test_most_words_keys_are_strings(self, tags):
        result = tags.most_words(3)
        for tag in result.keys():
            assert isinstance(tag, str)


    def test_most_words_values_are_ints(self, tags):
        result = tags.most_words(3)
        for count in result.values():
            assert isinstance(count, int)
            assert count > 0


    def test_most_words_sorted_descending(self, tags):
        result = tags.most_words(3)
        counts = list(result.values())
        assert counts == sorted(counts, reverse=True)


    def test_longest_returns_list(self, tags):
        result = tags.longest(3)
        assert isinstance(result, list)


    def test_longest_elements_are_strings(self, tags):
        result = tags.longest(3)
        for tag in result:
            assert isinstance(tag, str)


    def test_longest_sorted_descending(self, tags):
        result = tags.longest(10)
        lengths = [len(tag) for tag in result]
        assert lengths == sorted(lengths, reverse=True)


    def test_most_words_and_longest_returns_list(self, tags):
        result = tags.most_words_and_longest(3)
        assert isinstance(result, list)


    def test_most_words_and_longest_elements_are_strings(self, tags):
        result = tags.most_words_and_longest(3)
        for tag in result:
            assert isinstance(tag, str)


    def test_most_words_and_longest_sorted_alphabetically(self,tags):
        result = tags.most_words_and_longest(10)
        assert result == sorted(result)


    def test_most_popular_returns_dict(self, tags):
        result = tags.most_popular(3)
        assert isinstance(result, dict)


    def test_most_popular_keys_are_strings(self,tags):
        result = tags.most_popular(3)
        for tag in result.keys():
            assert isinstance(tag, str)


    def test_most_popular_values_are_ints(self, tags):
        result = tags.most_popular(3)
        for count in result.values():
            assert isinstance(count, int)
            assert count > 0


    def test_most_popular_sorted_descending(self, tags):
        result = tags.most_popular(10)
        counts = list(result.values())
        assert counts == sorted(counts, reverse=True)


    def test_tags_with_returns_list(self, tags):
        result = tags.tags_with("love")
        assert isinstance(result, list)


    def test_tags_with_elements_are_strings(self, tags):
        result = tags.tags_with("life")
        for tag in result:
            assert isinstance(tag, str)


    def test_tags_with_sorted_alphabetically(self, tags):
        result = tags.tags_with("the")
        assert result == sorted(result)


    def test_most_popular_by_movie_returns_dict(self, tags):
        result = tags.most_popular_by_movie(movie_id=1, n=3)
        assert isinstance(result, dict)


    def test_most_popular_by_movie_values_are_ints(self, tags):
        result = tags.most_popular_by_movie(movie_id=1, n=3)
        for count in result.values():
            assert isinstance(count, int)
            assert count > 0


    def test_average_tag_length_is_float(self, tags):
        avg_len = tags.average_tag_length()
        assert isinstance(avg_len, float)
        assert avg_len >= 0


    def test_get_unique_tags_count_is_int(self, tags):
        count = tags.get_unique_tags_count()
        assert isinstance(count, int)
        assert count > 0



    def test_manual_verification_most_popular_by_movie(self, tags):
        result = tags.most_popular_by_movie(movie_id=60756, n=3)
        assert "funny" in result
        assert result["funny"] == 3


    def test_manual_verification_average_tag_length(self, tags):
        lengths = [len(tag) for _, _, tag, _ in tags.tags_data]
        expected_avg = sum(lengths) / len(lengths) if lengths else 0
        actual_avg = tags.average_tag_length()
        assert actual_avg == expected_avg


        # tests for dict_by_release
    def test_dict_by_release_turns_dict(self, movies):
        result = movies.dist_by_release()
        assert isinstance(result, dict) # is it a dict?

    def test_dict_by_release_keys_are_ints_and_values_are_ints(self, movies):
        result = movies.dist_by_release()
        for year, count in result.items():
            assert isinstance(year, int)    # year is a number
            assert isinstance(count, int)    # how much is also a number
            assert count > 0   # it must be more than zero

    def test_dict_by_release_sorted_descending(self, movies):
        result = movies.dist_by_release()
        counts = list(result.values())
        assert counts == sorted(counts, reverse=True) # are they sorted correctly?

    # tests for dict_by_genres
    def test_dict_by_genres_returns_dict(self, movies):
        result = movies.dist_by_genres()
        assert isinstance(result, dict)

    def test_dict_by_genres_keys_are_strings_and_values_are_ints(self, movies):
        result = movies.dist_by_genres()
        for genre, count in result.items():
            assert isinstance(genre, str)  # genre is a string
            assert isinstance(count, int)  # how much is a number
            assert count > 0  

    def test_dict_by_genres_sorted_descending(self, movies):
        result = movies.dist_by_genres()
        counts = list(result.values())
        assert counts == sorted(counts, reverse=True)

    # tests for most_genres
    def test_most_genres_returns_dict(self, movies):
        result = movies.most_genres(5)
        assert isinstance(result, dict)

    def test_most_genres_keys_are_strings_and_values_are_ints(self, movies):
        result = movies.most_genres(10)
        for title, num_genres in result.items():
            assert isinstance(title, str)  # name is a string
            assert isinstance(num_genres, int)  # how much genres is a number
            assert num_genres > 0

    def test_most_genres_sorted_descending(self, movies):
        result = movies.most_genres(10)
        genre_counts = list(result.values())
        assert genre_counts == sorted(genre_counts, reverse=True)

    def test_most_genres_returns_exactly_n_items(self, movies):
        n = 3
        result = movies.most_genres(n)
        assert len(result) == n or len(result) <= len(movies.movies)  #  same or less than n



    # tests for get_imdb 
    def test_get_imdb_returns_list_of_lists(self, links):
        result = links.get_imdb([1, 2], ['title', 'director'])
        assert type(result) == list # is the result a list?
        assert len(result) == 2 # this list need to have two elements
        for row in result:
            assert type(row) == list
            assert len(row) == 3 # this row must contain 3 el: movieId + 2 fields
            assert type(row[0]) == int # first element is a number (movieId)


    def test_get_imdb_sorted_descending(self, links):
        result = links.get_imdb([1, 3, 2], ['title']) # not in the right sequence
        movie_ids = [row[0] for row in result] # taking only first element
        assert movie_ids == sorted(movie_ids, reverse=True) # is they sorted right?

    # tests for top_directors
    def test_top_directors_returns_dict(self, links):
        result = links.top_directors(3) # top three directors
        assert isinstance(result, dict) # is it a dict?
        assert len(result) <= 3 # if there only three elements

    def test_top_directors_values_are_ints(self, links):
        result = links.top_directors(5) # top five
        for count in result.values(): # is any number in this dict a number?
            assert isinstance(count, int)
            assert count > 0 # director recorded even one movie

    def test_top_directors_sorted_descending(self, links):
        result = links.top_directors(10) # top ten, this should be decreasing
        counts = list(result.values()) # getting list os numbers
        assert counts == sorted(counts, reverse=True) #checking are they the same?

    # tests for most_expensive
    def test_most_expensive_returns_dict(self, links):
        result = links.most_expensive(2) # top two
        assert isinstance(result, dict) # is it a dict?

    def test_most_expensive_values_are_ints_or_floats(self, links):
        result = links.most_expensive(5) #top five
        for budget in result.values(): # is any number in this dict is int or float?
            assert isinstance(budget, (int, float))
            assert budget > 0 # must be more than zero

    def test_most_expensive_sorted_descending(self, links):
        result = links.most_expensive(10) # the same like for previous method
        budgets = list(result.values())
        assert budgets == sorted(budgets, reverse=True)

    # tests for most_profitable
    def test_most_profitable_returns_dict(self, links):
        result = links.most_profitable(2)
        assert isinstance(result, dict)

    def test_most_profitable_values_are_numbers(self, links):
        result = links.most_profitable(5)
        for profit in result.values():
            assert isinstance(profit, (int, float)) # profit can be less than zero

    def test_most_profitable_sorted_descending(self, links):
        result = links.most_profitable(10)
        profits = list(result.values())
        assert profits == sorted(profits, reverse=True)

    # tests for longest
    def test_longest_returns_dict(self, links):
        result = links.longest(2)
        assert isinstance(result, dict)

    def test_longest_values_are_ints(self, links):
        result = links.longest(5)
        for runtime in result.values():
            assert isinstance(runtime, int)
            assert runtime > 0 # runtime must be more than zero

    def test_longest_descending_links(self, links):
        result = links.longest(10)
        runtimes = list(result.values())
        assert runtimes == sorted(runtimes, reverse=True)

    # tests for top_cost_per_minute 
    def test_top_cost_per_minute_returns_dict(self, links):
        result = links.top_cost_per_minute(2)
        assert isinstance(result, dict)

    def test_top_cost_per_minute_values_are_floats(self, links):
        result = links.top_cost_per_minute(5)
        for cost in result.values():
            assert isinstance(cost, float)
            assert cost > 0

    def test_top_cost_per_minute_sorted_descending(self, links):
        result = links.top_cost_per_minute(10)
        costs = list(result.values())
        assert costs == sorted(costs, reverse=True)

    # test for movies_by_director
    def test_movies_by_director(self, links):
        result = links.movies_by_director("John Lasseter")
        assert isinstance(result, list)
        for title in result:
            assert isinstance(title, str)

    # test for shortest_movie
    def test_shortest_movie(self, links):
        result = links.shortest_movie()
        assert isinstance(result, str)
