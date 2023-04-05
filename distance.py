"""
Tools for generating points from data, distances, and distance matrices.
"""
import numpy as np
from typing import List, Callable
import re
import math
from tqdm import tqdm
import os
finalarray = []

class Parser():
    def __init__(self):
        self.word_to_int = {}
        self.int_to_word = {}
        self.num_tweets = 0
        self.num_words = 0

    def clean_line(self, line: str) -> List[str]:
        """
        Strips and normalizes a tweet into a list of lowercase words.

        Args:
            line (str): a string representing a tweet

        Returns:
            List[str]: a normalized, lowercase list of the words in the tweet.
        """
        words = line.split('|')[2].lower().split(' ')
        words = [w for w in words if '/' not in w]
        words = [re.sub(r'[^a-zA-Z]', '', w) for w in words]
        return words

    def get_bag_of_words(self, filename: str, save_to: str="") -> np.ndarray:
        """
        Get the bag of words matrix for a given corpus.
        Every line in the corpus is represented as a row of length num_words,
        where num_words is the number of unique words in the corpus.
        Within each row, every element represents the number of times 
        a particular word appears in that line.

        Args:
            filename (str): the filename of the corpus
            save (str, optional): what filename to save the matrix to.

        Returns:
            np.ndarray: a matrix representing a bag of words embedding of the corpus.
        """
        if save_to and os.path.exists(save_to):
            print("Found previously generated matrix")
            return np.load(save_to)

        self.word_to_int = {}
        self.int_to_word = {}
        self.num_tweets = 0
        self.num_terms_total = 0
        self.num_words = 0
        sentences = []

        print("Gathering vocabulary...")
        with open(filename, 'r', encoding='utf8') as file:
            all_tweets = file.readlines()
            self.num_tweets = len(all_tweets)
            for tweet in tqdm(all_tweets):
                sentence = self.clean_line(tweet)
                sentences.append(sentence)
                # Add new words to vocabulary
                for word in sentence:
                    self.num_terms_total += 1
                    if word not in self.word_to_int:
                        self.word_to_int[word] = self.num_words
                        self.int_to_word[self.num_words] = word
                        self.num_words += 1
        
        print("Generating vectors...")
        vectors = np.zeros((self.num_tweets, self.num_words))
        for i in tqdm(range(self.num_tweets)):
            for word in sentences[i]:
                vectors[i][self.word_to_int[word]] += 1

        print("Generating text file...")
        txt_lines = [" ".join([self.int_to_word[i] for i in range(self.num_words)]) + "\n"]
        for t in range(self.num_tweets):
            txt_line = " ".join([str(num) for num in vectors[t]]) + "\n"
            txt_lines.append(txt_line)
        with open("tweet_matrix.txt", 'w') as file:
            file.writelines(txt_lines)

        if save_to:
            np.save(save_to, vectors)

        print("--- FEATURE MATRIX SUMMARY ---")
        print(f"Number of documents:    {self.num_tweets}")
        print(f"Number of term tokens:  {self.num_terms_total}")
        print(f"Number of unique terms: {self.num_words}")
        print(f"Avg terms per document: {round(self.num_terms_total/self.num_tweets, 2)}")

        return vectors

    def get_most_common_words(self, tweet_indices: np.ndarray, num_best_words: int=10) -> List[str]:
        """
        Get the most common non-stop words in a cluster.

        Args:
            tweet_indices (_type_): a numpy array of the indices of tweets in a cluster.
            num_best_words (int, optional): the number of most common words to get. Defaults to 10.

        Returns:
            List[str]: a list of the most common words in that cluster.
        """
        stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
        bag_of_words = self.get_bag_of_words("cnnhealth.txt")
        tweets = bag_of_words[tweet_indices]
        word_counts = np.sum(tweets, axis=0)
        indices = np.argsort(-word_counts)
        most_common_words = []
        for i in indices:
            word = self.int_to_word[i]
            if word not in stop_words:
                most_common_words.append(word)
                if len(most_common_words) >= num_best_words:
                    return most_common_words
        return most_common_words

        
    def get_distance_matrix(self, filename: str, distance_function: Callable, save_to: str="") -> np.ndarray:
        """
        Generate the distance matrix for the lines in a corpus.

        Args:
            filename (str): the name of the corpus to analyze
            distance_function (Callable): the function to be used as a distance function.
                This function should take two np.ndarray arguments of equal size and return a number.
            save (bool, optional): the filename to save the matrix to.

        Returns:
            np.ndarray: the distance matrix for the lines in the corpus.
        """
        if save_to and os.path.exists(save_to):
            print("Found previously generated matrix")
            return np.load(save_to)

        bag_of_words = self.get_bag_of_words(filename, None)
        distance_matrix = np.zeros((self.num_tweets, self.num_tweets))
        # generate upper triangle
        print("Generating distance matrix...")
        for i in tqdm(range(self.num_tweets - 1)):
            for j in range(i + 1, self.num_tweets):
                distance_matrix[i][j] = distance_function(bag_of_words[i], bag_of_words[j])
        # copy upper triangle to lower triangle
        distance_matrix = distance_matrix + np.transpose(distance_matrix)

        if save_to:
            np.save(save_to, distance_matrix)
        
        return distance_matrix

    def save_vocabulary(self):
        """
        Generate a text file with all words found in the corpus and their integer mappings.
        """
        lines = []
        for i in range(self.num_words):
            lines.append(f"{i} {self.int_to_word[i]}\n")
        with open("lexicon.txt", 'w') as file:
            file.writelines(lines)

def euclidean_distance(x, y):
  return np.linalg.norm(x - y)

def manhattan_distance(x, y):
  return np.sum(np.abs(x - y))