import streamlit as st
#from transformers import pipeline
#from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests
#import torch
import nltk
import random
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
nltk.download('punkt')

import openai

#openai.api_key = "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"
openai.api_key = "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"
#openai.api_key = "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"



API_TOKEN = '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'
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL_SUMMARY = "https://api-inference.huggingface.co/models/philschmid/bart-large-cnn-samsum"
API_URL_EMBEDDING = "https://api-inference.huggingface.co/models/nytimesrd/paraphrase-MiniLM-L6-v2"


def query(payload,url):
    data = json.dumps(payload)
    response = requests.request("POST", url, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def compute(txt):
    data = query(
        {
            "inputs": txt,
            "parameters": {"do_sample": False},
            "options" : {"wait_for_model":True}
        }, API_URL_SUMMARY
    )
    return data[0]['summary_text']
    #return data

def compute_embedding(txt):
    data = query(
        {
            "inputs": txt,
            "options" : {"wait_for_model":True}
        }, API_URL_EMBEDDING
    )
    return data

def get_common(txt1,txt2):
    sentences1 = sent_tokenize(txt1)
    sentences2 = sent_tokenize(txt1)
    # Compute embeddings for each sentence
    embeddings1 = np.array(compute_embedding(sentences1))
    embeddings2 = np.array(compute_embedding(sentences2))

    # Compute pairwise cosine similarities between the embeddings
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    # Find the most similar sentence pairs
    max_similarities = np.max(similarity_matrix, axis=1)
    most_similar_sentences = [(sentences1[i], sentences2[j], max_similarities[i]) for i, j in np.argwhere(similarity_matrix == max_similarities[:, None])]
    most_similar_sentences = sorted(most_similar_sentences, key=lambda x: x[2], reverse=True)
    most_similar_sentences = list(filter(lambda c : c[2] > 0.4,most_similar_sentences))
    most_similar_sentences = [a for a,b,c in most_similar_sentences]
    content = "\n".join(most_similar_sentences)
    return content

def get_keywords(text):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            #{"role": "system", "content": "You are a chatbot"},
            {"role": "user", "content": """Extract the top 10 key phrases from the following text. Give the output as a dictionary where the key is the phrase and the value is the corresponding float value given to the phrase in terms of capturing the context of the text.\n""" + text},
        ]
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    keywords = json.loads(result)
    return keywords


class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else [idx_min]

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, colors):
        """
        Draw the bubble plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : list
            Labels of the bubbles.
        colors : list
            Colors of the bubbles.
        """
        for i in range(len(self.bubbles)):
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)
            ax.text(*self.bubbles[i, :2], labels[i],
                    horizontalalignment='center', verticalalignment='center')

def plot_helper(result):
    keys = list(result.keys())
    values = list(result.values())
    values = [i*100 for i in values]
    number_of_colors = len(keys)

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(number_of_colors)]
    data = {
    'bigrams': keys
        ,

    'frequency': values,

    'color': color
    }   

    return data

def main():
    """Simple Login App"""

    st.title("Summarisation App")
    menu = ["Home", "Summarisation","Relevant content"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.title("This project is done by:")
        st.title("Deepthi Loser")
        st.title("Sathwik Acharya")
        st.slider("Rate how much you liked this project", min_value=0 , max_value=10)
        #view_all_users()


    elif choice == "Summarisation":
        st.subheader("Enter the text to be summarised in the space below")
        txt = st.text_area('Text to analyze')
        if st.button('Compute'):
            with st.spinner("Please wait for summary to load"):
                output = compute(txt)
                st.success("The summary is as follows")             
                st.markdown(output)
            
            with st.spinner("Please wait for chart to load"):
                keywords = get_keywords(txt)
                data = plot_helper(keywords)
                bubble_chart = BubbleChart(area=data['frequency'],
                           bubble_spacing=0.1)
                bubble_chart.collapse()
                fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
                fig.set_size_inches(9, 13, forward=True)
                bubble_chart.plot(
                    ax, data['bigrams'], data['color'])
                ax.axis("off")
                ax.relim()
                ax.autoscale_view()
                st.pyplot(fig)
    
    elif choice == "Relevant content":
        col1, col2 = st.columns(2)
        with col1:
            txt_1 = st.text_area('First Text to analyze')
        with col2:
            txt_2 = st.text_area('Second Text to analyze')
        
        if st.button('Compute'):
            with st.spinner("Please wait"):
                output = get_common(txt_1,txt_2)
                st.success("The common content is as follows")             
                st.markdown(output)

if __name__ == '__main__':
    main()
