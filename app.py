import streamlit as st
st.set_page_config(page_title="NLP", page_icon="📰")
from newspaper import Article
import re
import math
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
# Import libraries
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import newspaper
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
# from multi_emotion import multi_emotion
from pysentimiento import create_analyzer
import pandas as pd
import json
from textblob import TextBlob

# Define a lock
import threading
_lock = threading.Lock()

from langdetect import detect

# Streamlit app
st.title("📰Article Processor")
st.write("Enter the URLs of Articles to process:")
'''e.g. News Aticle, Wiki, etc'''

# Initialize with two URL input fields and checkboxes
# user_urls = [st.text_input("URL 1")]
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

user_urls = [st.text_input(
    "URL 1",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    placeholder="e.g. https://TOInews.com/article/trending/delhi-elections/",
    key="placeholder",
)]
# Maximum number of URLs allowed
max_urls = 5

# Add more input fields and checkboxes if needed, up to a maximum of 20
for i in range(max_urls - 1):
    if i < len(user_urls) and st.checkbox(f"URL {i + 2}", value=False):
        user_urls.append(st.text_input(f"URL {i + 2}"))


if st.button("Process URLs"):
    st.write("Please be patient for Amazing Results, it will take a few minutes")
    import nltk
    nltk.download('punkt')
    # Function to process a single URLl
    
    def process_url(url):
        try:
            import nltk
            print(nltk.data.path)
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()

            title = article.title
            text = article.text
            summary = article.summary
            keywords = article.keywords
            language = detect(text)  # Detect the language
    
            def read_article_text(text):
                # Tokenize the text into sentences
                sentences = sent_tokenize(text)
                return sentences

            def create_similarity_matrix(sentences, stop_words):
                # Create an empty similarity matrix
                similarity_matrix = np.zeros((len(sentences), len(sentences)))
                print("length of sentences is |",len(sentences))
            
                for index1 in range(len(sentences)):
                
                    for index2 in range(len(sentences)):
                        if index1 == index2: #ignore if both are same sentences
                            
                            continue 
                            
                        similarity_matrix[index1][index2] = sentence_similarity(sentences[index1], sentences[index2], stop_words)
                        
                    
                return similarity_matrix

            def sentence_similarity(sent1, sent2, stopwords=None):
                if stopwords is None:
                    stopwords = []
            
                sent1 = [w.lower() for w in sent1]
                sent2 = [w.lower() for w in sent2]
            
                all_words = list(set(sent1 + sent2))
                
                vector1 = [0] * len(all_words)
                vector2 = [0] * len(all_words)
            
                # build the vector for the first sentence
                for w in sent1:
                    if w in stopwords:
                        continue
                    
                    vector1[all_words.index(w)] += 1
                    
            
                # build the vector for the second sentence
                for w in sent2:
                    if w in stopwords:
                        continue
                    vector2[all_words.index(w)] += 1
                    
                    
                
                return (1 - cosine_distance(vector1, vector2))

            def text_summary(article_id, top_n=3):
                with open("en-stop-words.txt", "r") as sw:
                    stop_words = sw.read()
                summarize_text = []

                # Step 1 - Read text and split it to sentences.
                sentences = read_article_text(article_id)
                print("Sentences:", sentences)

                # Step 2 - Generate Similarity Matrix across sentences
                sentence_similarity_matrix = create_similarity_matrix(sentences, stop_words)

                # Step 3 - Rank sentences in similarity matrix
                sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
                scores = nx.pagerank(sentence_similarity_graph)
                print("Scores:", scores)
                # nx.draw(sentence_similarity_graph,with_labels=True)
                # Step 4 - Sort the rank and pick top sentences
                ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
                print("Ranked Sentence:", ranked_sentence)

                for i in range(top_n):
                    summarize_text.append("".join(ranked_sentence[i][1]))
                summarize_text = ". ".join(summarize_text)
                # Step 5 - Output the summarized text
                print("Summarize Text: \n", summarize_text)
                return summarize_text

            # def calculate_sentence_scores(sentences, stopwords):
            #     word_freq = Counter()
            #     sentence_scores = {}

            #     # Calculate word frequency for each sentence
            #     for sentence in sentences:
            #         words = word_tokenize(sentence.lower())
            #         word_freq.update(words)

            #     # Calculate sentence scores based on word frequency and position
            #     for sentence in sentences:
            #         words = word_tokenize(sentence.lower())
            #         score = 0
            #         for word in words:
            #             if word not in stopwords:
            #                 score += word_freq[word]
            #         sentence_scores[sentence] = score / len(words)

            #     return sentence_scores

            # def generate_summary(text, num_sentences=3):
            #     # Tokenize the text into sentences
            #     sentences = sent_tokenize(text)

            #     # Remove stopwords
            #     stop_words = set(stopwords.words('english'))

            #     # Calculate scores for each sentence
            #     sentence_scores = calculate_sentence_scores(sentences, stop_words)

            #     # Sort sentences by their scores in descending order
            #     sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

            #     # Select top 'num_sentences' sentences for the summary
            #     summary_sentences = sorted_sentences[:num_sentences]

            #     # Reorder the summary sentences based on their position in the original text
            #     summary_sentences = sorted(summary_sentences, key=lambda x: sentences.index(x[0]))

            #     # Combine the selected sentences into the final summary
            #     summary = ' '.join(sentence for sentence, score in summary_sentences)

            #     return summary
            return {
            "Title": title,
            "Article Text": text,
            "Article Summary": summary,
            "Article Keywords": keywords,
            "Language": language
        }
        except Exception as e:
            print(f"exception {e}")
            return None

    user_urls = [url for url in user_urls if url.strip()]  # Remove empty URLs
    user_news_data = []

    if not user_urls:
        st.write("Please enter valid URLs to proceed.")
    else:
        for url in user_urls:
            news_item = process_url(url)
            if news_item is not None:
                user_news_data.append(news_item)

        if not user_news_data:
            st.write("No valid URLs provided. Please enter valid URLs.")
        else:
            english_texts = []
            hindi_texts = []

            for news_item in user_news_data:
                if news_item['Language'] == 'en':
                    english_texts.append(news_item['Article Text'])
                elif news_item['Language'] == 'hi':
                    hindi_texts.append(news_item['Article Text'])

            if english_texts:
                # Process English texts here (e.g., create word cloud or bar chart)
                
                # Create word cloud or bar chart for English english_text
                # ...
                english_text = '\n'.join(news_item['Article Text'] for news_item in user_news_data)
                # from setup import analyzer, emotion_analyzer, hate_speech_analyzer
                # from transformers import AddedToken
                
                # # Define a custom hash function for tokenizers.AddedToken
                # def my_hash_func(token):
                #     try:
                #         return hash((token.ids, token.type_id))
                #     except AttributeError:
                #         # Handle cases where the token object is not as expected
                #         return hash(str(token))
                
                # @st.cache(allow_output_mutation=True, hash_funcs={AddedToken: my_hash_func})
                # def create_analyzers():
                #     return analyzer, emotion_analyzer, hate_speech_analyzer
                
                # analyzers = create_analyzers()
                
                # sentiment1 = analyzers[0].predict(english_text)
                # emotion1 = analyzers[1].predict(english_text)
                # hate_speech1 = analyzers[2].predict(english_text)
                




                TOInews = re.sub("[^A-Za-z" "]+", " ", english_text).lower()
                
                TOInews_tokens = TOInews.split(" ")

                with open("en-stop-words.txt", "r") as sw:
                    stop_words = sw.read()
                    
                stop_words = stop_words.split("\n")

                tokens = [w for w in TOInews_tokens if not w in stop_words]

                tokens_frequencies = Counter(tokens)

                # tokens_frequencies = tokens_frequencies.loc[tokens_frequencies.english_text != "", :]
                # tokens_frequencies = tokens_frequencies.iloc[1:]

                # Sorting
                tokens_frequencies = sorted(tokens_frequencies.items(), key = lambda x: x[1])

                # Storing frequencies and items in separate variables 
                frequencies = list(reversed([i[1] for i in tokens_frequencies]))
                words = list(reversed([i[0] for i in tokens_frequencies]))

                # Barplot of top 10 
                # import matplotlib.pyplot as plt
                
                
                # Create a figure and bar chart
                # Check if there are at least 11 unique words
                if len(words) < 11:
                    st.error("Error: At least 11 unique words are required for top tokens bar chart.")
                else:
                    # Create a figure and bar chart
                    with _lock:
                        plt.figure(1)
                        plt.bar(height=frequencies[0:11], x=range(0, 11), color=['red', 'green', 'black', 'yellow', 'blue', 'pink', 'violet'], width=0.6)
                        plt.title("Top 10 Tokens (Words)")
                        plt.grid(True)
                        # Customize the x-axis labels and rotation for visibility
                        plt.xticks(range(0, 11), words[0:11], rotation=45)
                        plt.xlabel("Tokens")
                        plt.ylabel("Count")

                        # Display the plot in Streamlit
                        st.pyplot(plt.figure(1), use_container_width=True)
                        

                ##########

                st.write("Please be patience for Amazing Results it will take a minute")
                # Joinining all the tokens into single paragraph 
                cleanstrng = " ".join(words)

                with _lock:
                    plt.figure(2)
                    wordcloud_ip = WordCloud(background_color = 'White', width = 2800, height = 2400).generate(cleanstrng)
                    plt.title("Normal Word Cloud")
                    plt.axis("off")
                    plt.grid(False)
                    plt.imshow(wordcloud_ip)
                    st.pyplot(plt.figure(2), use_container_width=True)


                #########################################################################################

                # positive words

                with open("en-positive-words.txt", "r") as pos:
                  poswords = pos.read().split("\n")
                
                # Positive word cloud
                # Choosing the only words which are present in positive words
                # Positive word cloud
                # Choosing the only words which are present in positive words
                pos_tokens = " ".join ([w for w in TOInews_tokens if w in poswords])

                with _lock:
                    plt.figure(3)
                    if pos_tokens:
                        # wordcloud_positive = WordCloud(background_color='White', width=1800, height=1400).generate(" ".join(pos_tokens))
                        wordcloud_positive = WordCloud(background_color = 'White', width = 1800, height = 1400).generate(pos_tokens)
                        plt.title("Positive Word Cloud")
                        plt.axis("off")
                        plt.grid(False)
                        plt.imshow(wordcloud_positive)
                        st.pyplot(plt.figure(3), use_container_width=True)
                    else:
                        plt.text(0.5, 0.5, "No positive words found", horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
                        plt.axis("off")
                        plt.grid(False)
                        st.pyplot(plt.figure(3), use_container_width=True)

                # Negative words
               
                with open("en-negative-words.txt", "r") as neg:
                  negwords = neg.read().split("\n")
                # Negative word cloud
                # Choosing the only words which are present in negwords
                neg_tokens = " ".join ([w for w in TOInews_tokens if w in negwords])
                with _lock:
                    plt.figure(4)
                    if neg_tokens:
                        wordcloud_negative = WordCloud(background_color = 'black', width = 1800, height=1400).generate(neg_tokens)
                        plt.title("Negative Word Cloud")
                        plt.axis("off")
                        plt.grid(False)
                        plt.imshow(wordcloud_negative)
                        st.pyplot(plt.figure(4), use_container_width=True)
                    else:
                        plt.text(0.5, 0.5, "No negative words found", horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
                        plt.axis("off")
                        plt.grid(False)
                        st.pyplot(plt.figure(4), use_container_width=True)
                #########################################################################################
                
                
                # Word cloud with 2 words together being repeated

                # Extracting n-grams using TextBlob

                bigrams_list = list(nltk.bigrams(tokens))
                dictionary2 = [' '.join(tup) for tup in bigrams_list]

                # Using count vectorizer to view the frequency of bigrams
                
                vectorizer = CountVectorizer(ngram_range = (2, 2))
                bag_of_words = vectorizer.fit_transform(dictionary2)
                v1 = vectorizer.vocabulary_

                sum_words = bag_of_words.sum(axis = 0)
                words_freq = [(word, sum_words[0, idx]) for word, idx in v1.items()]
                words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

                words_dict = dict(words_freq[:100])
                with _lock:
                    plt.figure(5)
                    wordcloud_2 = WordCloud(background_color = 'black', width = 1800, height = 1400)                 
                    wordcloud_2.generate_from_frequencies(words_dict)
                    plt.title("Bi-Gram based on Frequency")
                    plt.axis("off")
                    plt.grid(False)
                    plt.imshow(wordcloud_2)
                    st.pyplot(plt.figure(5), use_container_width=True)
                ##############################################################################################
                
                # Word cloud with 2 words together being repeated
                
                # Extracting n-grams using TextBlob

                bigrams_list2 = list(nltk.trigrams(tokens))
                dictionary3 = [' '.join(tup) for tup in bigrams_list2]

                # Using count vectorizer to view the frequency of bigrams
                
                vectorizer1 = CountVectorizer(ngram_range = (3, 3))
                bag_of_words1 = vectorizer1.fit_transform(dictionary3)
                v2 = vectorizer1.vocabulary_

                sum_words1 = bag_of_words1.sum(axis = 0)
                words_freq1 = [(word1, sum_words1[0, idx1]) for word1, idx1 in v2.items()]
                words_freq1 = sorted(words_freq1, key = lambda x: x[1], reverse = True)

                words_dict1 = dict(words_freq1[:100])
                with _lock:
                    plt.figure(6)
                    wordcloud_3 = WordCloud(background_color = 'black', width = 1800, height = 1400)                  
                    wordcloud_3.generate_from_frequencies(words_dict1)
                    plt.title("Tri-Gram based on Frequency")
                    plt.grid(False)
                    plt.axis("off")
                    plt.imshow(wordcloud_3)
                    st.pyplot(plt.figure(6), use_container_width=True)

                # eqn shift 1
                pattern = "[^A-Za-z.]+"

                # Perform english_text preprocessing without removing full stops
                sen = re.sub(pattern, " ", english_text).lower()

                # SENTANCE Tokenizer
                sen_t = sen.split(".")


                # Create a DataFrame with the sentences as lists
                df = pd.DataFrame(sen_t)

                # Display the DataFrame
                print(df)

                df.columns = ['english_text']
                

                # Number of words
                df['number_of_words'] = df['english_text'].apply(lambda x : len(TextBlob(x).words))

                # Detect presence of wh words
                wh_words = set(['why', 'who', 'which', 'what', 'where', 'when', 'how'])
                df['are_wh_words_present'] = df['english_text'].apply(lambda x : True if len(set(TextBlob(str(x)).words).intersection(wh_words)) > 0 else False)


                # Polarity
                df['polarity'] = df['english_text'].apply(lambda x : TextBlob(str(x)).sentiment.polarity)

                # Subjectivity
                df['subjectivity'] = df['english_text'].apply(lambda x : TextBlob(str(x)).sentiment.subjectivity)

                
                # Calculate the average number of words
                average_words = df['number_of_words'].mean()

                # Calculate the percentage of sentences that have WH words
                average_wh_presence = (df['are_wh_words_present'].sum() / len(df)) * 100

                # Calculate the average polarity
                average_polarity = df['polarity'].mean()

                # Calculate the average subjectivity
                average_subjectivity = df['subjectivity'].mean()

                # Display the calculated averages
                print("Average Number of Words:", average_words)
                print("Average Percentage of Sentences with WH Words:", average_wh_presence)
                print("Average Polarity:", average_polarity)
                print("Average Subjectivity:", average_subjectivity)

                # Create a DataFrame to store the results
                results_df = pd.DataFrame({
                    'Metric': ['Average Number of Words', 'Average Percentage of Sentences with WH Words', 'Average Polarity', 'Average Subjectivity'],
                    'Value': [average_words, average_wh_presence, average_polarity, average_subjectivity]
                })

                # Display the results DataFrame
                print(results_df)
                # eqn shift 1

                # results_df = pd.DataFrame(results_df)
                # Set a Seaborn color palette for styling
                
                
                # Streamlit app
                st.subheader("Sentiment Analysis Dataframe")
                
                # Display the DataFrame using Seaborn styling
                st.table(results_df)
                # Open the file in read mode and read its content into a variable
                
                # emo_in_txt = english_text
                
                # result = multi_emotion.predict([emo_in_txt])


                # # Extract english_text, labels, and probabilities
                # text1 = result[0]['english_text']
                # labels = result[0]['pred_label'].split(',')
                # probabilities = json.loads(result[0]['probability'])

                # # Split the english_text into words
                # words_e = text1.split()

                # # Create a list of dictionaries to store emotions for each word
                # emotions_list = []

                # for word_e, prob in zip(words_e, probabilities):
                #     emotions = {'word': word_e}
                #     emotions.update(prob)
                #     emotions_list.append(emotions)

                # print(emotions_list)


                # label_1 = [key for item in probabilities for key in item.keys()] # True

                # print(label_1) #TRUE

                # # Create a DataFrame to capture emotions for each word
                # emotions_df = pd.DataFrame(emotions_list)

                # # Print the DataFrame
                # print(emotions_df)
                # # Now you have a DataFrame with emotions for each word

                # # Assuming you have 'Happy', 'Angry', 'Surprise', 'Sad', and 'Fear' columns in emotions_df
                # # You can now perform the additional operations as requested:
                # tokens_df = pd.DataFrame(words_e, columns=['words'])

                # emp_emotions = pd.concat([tokens_df, emotions_df], axis=1)


                # emotions_to_plot = label_1
                # sum_emotions = emp_emotions[emotions_to_plot].sum()

                # # Plot the summed emotions
                # with _lock:
                #     plt.figure(7)
                #     sum_emotions.plot(kind='bar', color=['pink', 'orange', 'blue', 'yellow', 'green', 'purple', 'red', 'cyan', 'magenta', 'lime'])
                #     plt.title('Sum of Emotions for the english_text')
                #     plt.xlabel('Emotions')
                #     plt.ylabel('Sum')
                #     plt.show()
                #     st.pyplot(plt.figure(7), use_container_width=True)
                # print(emp_emotions.head(20))
                # from setup import analyzer, emotion_analyzer, hate_speech_analyzer
                # analyzer = create_analyzer(task="sentiment", lang="en")
                
                # from transformers import AddedToken
                
                # # Define a custom hash function for tokenizers.AddedToken
                # def my_hash_func(token):
                #     try:
                #         return hash((token.ids, token.type_id))
                #     except AttributeError:
                #         # Handle cases where the token object is not as expected
                #         return hash(str(token))
                
                # @st.cache_resource(hash_funcs={AddedToken: my_hash_func})
                # def get_analyzers():
                #     from setup import analyzer, emotion_analyzer, hate_speech_analyzer
                #     return analyzer, emotion_analyzer, hate_speech_analyzer
                
                # from my_module import get_analyzers
                # # Load analyzers
                # analyzers = get_analyzers()
                
                @st.cache_resource()
                def get_analyzers():
                    from setup import analyzer, emotion_analyzer, hate_speech_analyzer
                    return analyzer, emotion_analyzer, hate_speech_analyzer
                analyzers = get_analyzers()
                # Now you can use the analyzers for text analysis
                sentiment1 = analyzers[0].predict(english_text)
                emotion1 = analyzers[1].predict(english_text)
                hate_speech1 = analyzers[2].predict(english_text)
                
                # sentiment1 = analyzer.predict(emo_in_txt)
                st.subheader("Sentiment Analysis")
                st.write(sentiment1)
                sentiment_output = sentiment1.output
                probas_sentiment = sentiment1.probas
                NEU = probas_sentiment.get("NEU")
                POS = probas_sentiment.get("POS")
                NEG = probas_sentiment.get("NEG")
                

                # Create labels and values for the pie chart
                labels = ['NEU', 'POS', 'NEG']
                values = [NEU, POS, NEG]
                colors = ['blue', 'green', 'red']
                
                with _lock:
                    # Create a figure with the figure number 7
                    plt.figure(7, figsize=(6, 6))
                    
                    # Create a pie chart with custom colors
                    wedges, _ = plt.pie(values, colors=colors, startangle=90)
                    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
                    
                    # Create a legend with labels and values
                    legend_labels = [f"{label}: {value:.1%}" for label, value in zip(labels, values)]
                    plt.legend(wedges, legend_labels, title="Sentiments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                    plt.show()
                    st.pyplot(plt.figure(7), use_container_width=True)

                st.write("Sentiment Output:", sentiment_output)
                st.write("Probas Sentiment:")
                st.write("NEU:", NEU)
                st.write("POS:", POS)
                st.write("NEG:", NEG)
                
                
                # emotion_analyzer = create_analyzer(task="emotion", lang="en")
                # emotion1 = emotion_analyzer.predict(emo_in_txt)
                st.subheader("Emotion Analysis")
                st.write(emotion1)
                emotion_output = emotion1.output
                probas_emotion = emotion1.probas
                others = probas_emotion.get("others")
                joy = probas_emotion.get("joy")
                disgust = probas_emotion.get("disgust")
                fear = probas_emotion.get("fear")
                sadness = probas_emotion.get("sadness")
                surprise = probas_emotion.get("surprise")
                anger = probas_emotion.get("anger")
                

                # Create a dictionary for the emotion probabilities
                emotions101 = {
                    "Others": others,
                    "Joy": joy,
                    "Disgust": disgust,
                    "Fear": fear,
                    "Sadness": sadness,
                    "Surprise": surprise,
                    "Anger": anger
                }
                # Extract emotion labels and probabilities
                emotions = emotions101.keys()
                probabilities = emotions101.values()
                
                # Create a bar plot
                with _lock:
                    plt.figure(8,figsize=(10, 6))
                    plt.bar(emotions, probabilities, color=['blue', 'green', 'red', 'purple', 'orange', 'pink'])
                    plt.xlabel("Emotion")
                    plt.ylabel("Probability")
                    plt.title("Emotion Probabilities")
                    plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
                    plt.show()
                    st.pyplot(plt.figure(8), use_container_width=True)

                st.write("Emotion Output:", emotion_output)
                st.write("Probas Emotion:")
                st.write("Others:", others)
                st.write("Joy:", joy)
                st.write("Disgust:", disgust)
                st.write("Fear:", fear)
                st.write("Sadness:", sadness)
                st.write("Surprise:", surprise)
                st.write("Anger:", anger)
                # Show the plot
                
                # st.bar_chart(emotions101)
                # with _lock:
                #     plt.figure(8)
                #     plt.barh(list(emotions101.keys()), list(emotions101.values()))
                #     plt.xlabel('Probability')
                #     plt.title('Emotion Analysis')
                #     st.pyplot(plt.figure(8), use_container_width=True)
                
                # hate_speech_analyzer = create_analyzer(task="hate_speech", lang="en")

                # hate_speech1 =  hate_speech_analyzer.predict(emo_in_txt)
                st.subheader("Hate Speech Analysis")
                st.write(hate_speech1)
                hate_speech1_output = hate_speech1.output
                probas_hate_speech1 = hate_speech1.probas
                # Extract the values
                hateful = probas_hate_speech1.get("hateful")
                targeted = probas_hate_speech1.get("targeted")
                aggressive = probas_hate_speech1.get("aggressive")
                
             
                
                # Create a dictionary for the hate speech probabilities
                hate_speech = {
                    "Hateful": hateful,
                    "Targeted": targeted,
                    "Aggressive": aggressive
                }
                
                # Extract hate speech labels and probabilities
                labels = hate_speech.keys()
                probs = hate_speech.values()
                
                # Create a bar plot
                with _lock:
                    plt.figure(9,figsize=(10, 6))
                    plt.bar(labels, probs, color=['red', 'green', 'blue'])
                    plt.xlabel("Category")
                    plt.ylabel("Probability")
                    plt.title("Hate Speech Probabilities")
                    plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
                    # Show the plot
                    plt.show()
                    st.pyplot(plt.figure(9), use_container_width=True) 
                    st.write("Hate Speech Output:", hate_speech1_output)
                
                st.write("Probas:")
                st.write("Hateful:", hateful)
                st.write("Targeted:", targeted)
                st.write("Aggressive:", aggressive) 

                for idx, news_item in enumerate(user_news_data, start=1):
                    st.subheader(f"Article {idx}: {news_item['Title']}")
                    st.write("Article Text:")
                    st.write(news_item['Article Text'])
                    st.write("Article Summary:")
                    st.write(news_item['Article Summary'])
                    st.write("Article Summery2")
                    st.write(news_item['summary 2.0'])
                    st.write("Article Keywords:")
                    st.write(', '.join(news_item['Article Keywords']))
                    st.markdown("---")    
                st.balloons()
                

            else:
                st.write("Only English Language is Supported from now... 🥲")
 