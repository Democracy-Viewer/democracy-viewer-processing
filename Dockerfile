FROM amazonlinux:latest

# Set the working directory within docker image
WORKDIR /usr/src/app

# Add all files
ADD . .

# Install Python dependencies
RUN yum install -y python3.11 python3.11-pip git
RUN python3.11 -m pip install -r requirements.txt

# Install NLTK packages
RUN python3.11 -m nltk.downloader stopwords punkt

# Install SpaCy language models with explicit versions
### Chinese
RUN python3.11 -m pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.7.0/zh_core_web_sm-3.7.0-py3-none-any.whl
### Dutch
RUN python3.11 -m pip install https://github.com/explosion/spacy-models/releases/download/nl_core_news_sm-3.7.0/nl_core_news_sm-3.7.0-py3-none-any.whl
### English
RUN python3.11 -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
### French
RUN python3.11 -m pip install https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.7.0/fr_core_news_sm-3.7.0-py3-none-any.whl
### German
RUN python3.11 -m pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.7.0/de_core_news_sm-3.7.0-py3-none-any.whl
### Greek
RUN python3.11 -m pip install https://github.com/explosion/spacy-models/releases/download/el_core_news_sm-3.7.0/el_core_news_sm-3.7.0-py3-none-any.whl
### Italian
RUN python3.11 -m pip install https://github.com/explosion/spacy-models/releases/download/it_core_news_sm-3.7.0/it_core_news_sm-3.7.0-py3-none-any.whl
### Portuguese
RUN python3.11 -m pip install https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.7.0/pt_core_news_sm-3.7.0-py3-none-any.whl
### Russian
RUN python3.11 -m pip install https://github.com/explosion/spacy-models/releases/download/ru_core_news_sm-3.7.0/ru_core_news_sm-3.7.0-py3-none-any.whl
### Spanish
RUN python3.11 -m pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.7.0/es_core_news_sm-3.7.0-py3-none-any.whl

# Third party language models
### Latin
# RUN python3.11 -m pip install https://huggingface.co/latincy/la_core_web_sm/resolve/main/la_core_web_sm-any-py3-none-any.whl