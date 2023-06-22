import glob
import os
from datetime import datetime

import numpy as np
import openai
from dotenv import dotenv_values
from langchain import OpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

config = dotenv_values('.env')
openai.api_key = config['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

# Paths and patterns
local_repo_path = '/Users/adrianashymoniak/projects/azure-search-openai-demo/**'
file_patterns = ['*.json', '*.txt', '*.py', '*.md']


class Analyzer:
    main_prompt = """
    Firstly, give the following text an informative title. 
    Then, on a new line, write a 75-100 word summary of the following text:
    {text}
    Return your answer in the following format:
    Title | Summary...
    e.g. 
    Why Artificial Intelligence is Good | AI can make humans more productive by 
    automating many repetitive processes.
    TITLE AND CONCISE SUMMARY:
    """

    def __init__(self, file, content):
        self.map_llm = OpenAI(temperature=0, model_name='text-davinci-003')
        self.file = file
        self.content = content

    @staticmethod
    def get_files_from_dir(dir_path: str, patterns: list):
        """
        Function to get files from local directory

        """
        files_list = []
        for pattern in patterns:
            files_list.extend(glob.glob(dir_path + '/' + pattern, recursive=True))
        return files_list

    @staticmethod
    def read_files(file_paths):
        """
        Function to read content of the files
        """

        contents_dict = {}
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                contents_dict[file_path] = f.read()
        return contents_dict

    @staticmethod
    def get_chunks_from_text(text, num_chunks=10):
        """
        Function to break a large text into chunks
        """

        words = text.split()
        words_per_chunk = len(words) // num_chunks
        chunks_list = []
        for i in range(0, len(words), words_per_chunk):
            chunk = ' '.join(words[i:i + words_per_chunk])
            chunks_list.append(chunk)
        return chunks_list

    def summarize_chunks(self, chunks_list, template):
        """
        Function to summarize chunks_list using OpenAI
        """

        llm_chain = LLMChain(llm=self.map_llm, prompt=template)
        summaries = []
        for chunk in chunks_list:
            chunk_summary = llm_chain.apply([{'text': chunk}])
            summaries.append(chunk_summary)
        return summaries

    @staticmethod
    def create_similarity_matrix(chunks_list):
        """
        Function to calculate similarity matrix
        """

        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform(
            [' '.join(chunk.split()[:200]) for chunk in chunks_list])
        return cosine_similarity(vectors)

    @staticmethod
    def get_topics(similarity_matrix_, num_topics=5):
        """
        Get the topics from the similarity matrix
        """
        distances = 1 - similarity_matrix_
        kmeans = KMeans(n_clusters=num_topics).fit(distances)
        clusters = kmeans.labels_
        chunk_topics = [np.where(clusters == i)[0] for i in range(num_topics)]
        return chunk_topics

    @staticmethod
    def parse_title_summary_results(results):
        """
        Function to parse title and summary results
        """

        outputs = []
        for result in results:

            result = result.replace('\n', '')
            if '|' in result:
                processed = {'title': result.split('|')[0],
                             'summary': result.split('|')[1][1:]
                             }
            elif ':' in result:
                processed = {'title': result.split(':')[0],
                             'summary': result.split(':')[1][1:]
                             }
            elif '-' in e:
                processed = {'title': result.split('-')[0],
                             'summary': result.split('-')[1][1:]
                             }
            else:
                processed = {'title': '',
                             'summary': result
                             }
            outputs.append(processed)
        return outputs

    def summarize_stage(self, chunks_list, topics_list):
        """
        Function to summarize the stage
        """

        print(f'Start time: {datetime.now()}')

        # Prompt to get title and summary for each topic
        prompt = """Write a detailed summary on the structure of the provided 
        content which contains code from selected files from a Github repository, which 
        deploys a chatbot system in Microsoft Azure. Please list all necessary details 
        which can be extrapolated later to specific guidelines how to reverse engineer 
        the repository. I am specifically looking for answers on: 
        i. the specific steps to deploy this resource? Please list all the files that 
        contain the specific tasks that automate the deployment!
        ii. relevant files and code sections that I need to alter in case I want to 
        adjust the overall tool of the repository for my use case. Please list all 
        files and name the code sections.
        iii. the detailed steps that need to be performed in order to adjust this 
        repository as a project template for customized deployments.: 
        {text}
        """

        map_prompt = PromptTemplate(template=prompt,
                                    input_variables=["text"])

        # Define the LLMs
        map_llm_chain = LLMChain(llm=self.map_llm, prompt=map_prompt)

        summaries = []
        for i in range(len(topics_list)):
            topic_summaries = []
            for topic in topics_list[i]:
                map_llm_chain_input = [{'text': chunks_list[topic]}]
                # Run the input through the LLM chain (works in parallel)
                map_llm_chain_results = map_llm_chain.apply(map_llm_chain_input)
                stage_1_outputs = Analyzer.parse_title_summary_results(
                    [e['text'] for e in map_llm_chain_results])
                # Split the titles and summaries
                topic_summaries.append(stage_1_outputs[0]['summary'])
            # Concatenate all summaries of a topic
            summaries.append(' '.join(topic_summaries))

        print(f'Stage done time {datetime.now()}')

        return summaries

    @staticmethod
    def get_prompt_template(template):
        return PromptTemplate(template=template,
                              input_variables=['text'])

    def analyze_file(self):
        print(f'Processing {self.file}...')

        chunks = Analyzer.get_chunks_from_text(self.content)

        # Summarize chunks
        chunk_summaries = (
            self.summarize_chunks(
                chunks,
                self.get_prompt_template(Analyzer.main_prompt))
        )

        # Create similarity matrix
        similarity_matrix = Analyzer.create_similarity_matrix(chunks)

        # Get topics
        topics = Analyzer.get_topics(similarity_matrix)

        # Summarize stage
        stage_summary = self.summarize_stage(chunk_summaries, topics)

        print(f'Summary for {self.file}:\n{stage_summary}\n')


if __name__ == "__main__":
    """
    Main script
    """

    # Fetch files
    files = Analyzer.get_files_from_dir(local_repo_path, file_patterns)

    # Iterate over files and process
    for _file, _content in Analyzer.read_files(files).items():
        code_analyzer = Analyzer(_file, _content)
        code_analyzer.analyze_file()

    print('All files processed.')
