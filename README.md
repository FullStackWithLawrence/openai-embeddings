# Hybrid Search Retriever

A Hybrid Search and Augmented Generation prompting solution using Python [OpenAI](https://openai.com/) API embeddings sourced from [Pinecone](https://docs.pinecone.io/docs/python-client) vector database indexes and managed by [LangChain](https://www.langchain.com/).

Implements the following:

- a command-line pdf loader program that extracts text, vectorizes, and
  loads into a Pinecone dot product vector database that is dimensioned to match OpenAI embeddings.

- a hybrid search retriever that locates relevant documents from the vector database and includes these in OpenAI prompts.

Features:

- automated PDF document loader
- Seamless OpenAI embeddings using Langchain
- PineconeIndex helper class that fully manages the lifecycle of Pinecone vector database indexes
- Quickstart: `make init`
- Parameterized modules
- 20+ automated unit tests
- Preconfigured for seamless integration between OpenAI and Pinecone

See:

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)
- [What is a Vector Database?](https://www.pinecone.io/learn/vector-database/)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)
- [LanchChain Caching](https://python.langchain.com/docs/modules/model_io/llms/llm_caching)

## Installation

```console
git clone https://github.com/lpm0073/netec-llm.git
cd netec-llm
make init
source venv/bin/activate
```

You'll also need to add your api keys to the .env file in the root of the repo.

- Get your [OpenAI API key](https://platform.openai.com/api-keys)
- Get your [Pinecone API Key](https://app.pinecone.io/)

```console
OPENAI_API_ORGANIZATION=PLEASE-ADD-ME
OPENAI_API_KEY=PLEASE-ADD-ME
PINECONE_API_KEY=PLEASE-ADD-ME
```

## Usage

```console
# command-line help
python3 -m models.ssm -h

# example 1 - generic assistant
python3 -m models.examples.prompt "your are a helpful assistant" "explain why LangChain is so popular for generative AI development"

# example 2 - untrained assistant with expertise on Mexico City businesses
python3 -m models.examples.prompt "You are an expert on businesses located in Mexico City. You provide concise answers of 100 words or less." "What kinds of training does Netec offer?"

# example 3 - prompted assistant
python3 -m models.examples.training_services "Microsoft certified Azure AI engineer associate"

# example 4 - prompted assistant
python3 -m models.examples.training_services_oracle "Oracle database administrator"

# example 5 - Retrieval Augmented Generation
python3 -m models.examples.pinecone_init
python3 -m models.examples.load "./data/"
python3 -m models.examples.rag "What analytics and accounting courses does Wharton offer?"
```

## Configuration defaults

Set these as environment variables on the command line, or in a .env file that should be located in the root of the repo.

```console
# OpenAI API
OPENAI_API_ORGANIZATION=PLEASE-ADD-ME
OPENAI_API_KEY=PLEASE-ADD-ME
OPENAI_CHAT_MAX_RETRIES=3
OPENAI_CHAT_MODEL_NAME=gpt-3.5-turbo
OPENAI_CHAT_TEMPERATURE=0.0
OPENAI_PROMPT_MODEL_NAME=text-davinci-003

# Pinecone API
PINECONE_API_KEY=PLEASE-ADD-ME
PINECONE_DIMENSIONS=1536
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=rag
PINECONE_METRIC=dotproduct
PINECONE_VECTORSTORE_TEXT_KEY=lc_id

# This package
DEBUG_MODE=False
```

## Contributing

This project uses a mostly automated pull request and unit testing process. See the resources in .github for additional details. You additionally should ensure that pre-commit is installed and working correctly on your dev machine by running the following command from the root of the repo.

```console
pre-commit run --all-files
```

Pull requests should pass these tests before being submitted:

```console
make test
```

### Developer setup

```console
git clone https://github.com/lpm0073/automatic-models.git
cd automatic-models
make init
make activate
```

### Github Actions

Actions requires the following secrets:

```console
PAT: {{ secrets.PAT }}  # a GitHub Personal Access Token
OPENAI_API_ORGANIZATION: {{ secrets.OPENAI_API_ORGANIZATION }}
OPENAI_API_KEY: {{ secrets.OPENAI_API_KEY }}
PINECONE_API_KEY: {{ secrets.PINECONE_API_KEY }}
PINECONE_ENVIRONMENT: {{ secrets.PINECONE_ENVIRONMENT }}
PINECONE_INDEX_NAME: {{ secrets.PINECONE_INDEX_NAME }}
```
