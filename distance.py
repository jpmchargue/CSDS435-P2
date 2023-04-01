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

    def get_bag_of_words(self, filename: str, save: bool=True) -> np.ndarray:
        """
        Get the bag of words matrix for a given corpus.
        Every line in the corpus is represented as a row of length num_words,
        where num_words is the number of unique words in the corpus.
        Within each row, every element represents the number of times 
        a particular word appears in that line.

        Args:
            filename (str): the filename of the corpus
            save (bool, optional): whether to save the resulting matrix. Defaults to True.

        Returns:
            np.ndarray: _description_
        """
        if os.path.exists(filename):
            print("Found previously generated matrix")
            return np.load(filename)

        self.word_to_int = {}
        self.int_to_word = {}
        self.num_tweets = 0
        self.num_words = 0
        sentences = []

        print("Gathering vocabulary...")
        with open(filename, 'r') as file:
            all_tweets = file.readlines()
            self.num_tweets = len(all_tweets)
            for tweet in tqdm(all_tweets):
                sentence = self.clean_line(tweet)
                sentences.append(sentence)
                # Add new words to vocabulary
                for word in sentence:
                    if word not in self.word_to_int:
                        self.word_to_int[word] = self.num_words
                        self.int_to_word[self.num_words] = word
                        self.num_words += 1
        
        print("Generating vectors...")
        vectors = np.zeros((self.num_tweets, self.num_words))
        for i in tqdm(range(self.num_tweets)):
            for word in sentences[i]:
                vectors[i][self.word_to_int[word]] += 1

        if save:
            np.save("bag_of_words.npy", vectors)

        return vectors
        
    def get_distance_matrix(self, filename: str, distance_function: Callable, save: bool=True) -> np.ndarray:
        """
        Generate the distance matrix for the lines in a corpus.

        Args:
            filename (str): the name of the corpus to analyze
            distance_function (Callable): the function to be used as a distance function.
                This function should take two np.ndarray arguments of equal size and return a number.
            save (bool, optional): whether to save the resulting matrix. Defaults to True.

        Returns:
            np.ndarray: the distance matrix for the lines in the corpus.
        """
        if os.path.exists(filename):
            print("Found previously generated matrix")
            return np.load(filename)

        bag_of_words = self.get_bag_of_words(filename, save)
        distance_matrix = np.zeros((self.num_tweets, self.num_tweets))
        # generate upper triangle
        print("Generating distance matrix...")
        for i in tqdm(range(self.num_tweets - 1)):
            for j in range(i + 1, self.num_tweets):
                distance_matrix[i][j] = distance_function(bag_of_words[i], bag_of_words[j])
        # copy upper triangle to lower triangle
        distance_matrix = distance_matrix + np.transpose(distance_matrix)

        if save:
            np.save("distance_matrix.npy", distance_matrix)
        
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
  return np.sum(x - y)

parser = Parser()
dm = parser.get_distance_matrix("cnnhealth.txt", euclidean_distance)
parser.save_vocabulary()

