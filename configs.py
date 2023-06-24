import os

import openai
import requests
from dotenv import dotenv_values


class Config:
    def __init__(self):
        config = dotenv_values('.env')
        openai.api_key = config['OPENAI_API_KEY']
        os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

        # Paths and patterns
        self.dir_path = config['LOCAL_REPO_PATH']
        self.file_patterns = ['*.json', '*.txt', '*.py', '*.md']
        self.is_local = True
        if config['LOCAL_REPO'] == 'False':
            self.is_local = False
        self.github_repo = config['GITHUB_REPO']

        repo_metadata = self.github_repo.split("github.com/")[-1]
        self.owner, self.repo_name = repo_metadata.split("/")
        response = requests.get(f"https://api.github.com/repos/"
                                f"{self.owner}/{self.repo_name}")

        self.default_branch = 'master'
        if response.status_code == 200:
            repo_info_endpoint = f"https://api.github.com/repos/{self.owner}/{self.repo_name}"
            self.default_branch = (
                requests.get(repo_info_endpoint).json()['default_branch']
            )

