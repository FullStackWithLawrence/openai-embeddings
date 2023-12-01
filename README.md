# Hybrid Search Retriever

A Python [OpenAI](https://openai.com/) / [LangChain](https://www.langchain.com/) / [Pinecone](https://docs.pinecone.io/docs/python-client) proof of concept Retrieval Augmented Generation (RAG) model using PDF documents as the embeddings data source.

Implements the following:

- a command-line pdf loader program that extracts text, vectorizes, and
  loads into a Pinecone dot product vector database that is dimensioned to match OpenAI embeddings.

- a hybrid search retriever that locates relevant documents from the vector database and includes these in OpenAI prompts.

See:

- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)
- [LanchChain Caching](https://python.langchain.com/docs/modules/model_io/llms/llm_caching)

## Installation

```console
git clone https://github.com/lpm0073/netec-llm.git
cd netec-llm
make init
make activate
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

# example 5 - Load PDF documents
python3 -m models.examples.load "./data/"

# example 6 - Retrieval Augmented Generation
python3 -m models.examples.rag "What is Accounting Based Valuation?"
```

## Requirements

- OpenAI API key
- Pinecone API key

```console
export OPENAI_API_ORGANIZATION=SET-ME-PLEASE
export OPENAI_API_KEY=SET-ME-PLEASE
export PINECONE_API_KEY=SET-ME-PLEASE
export PINECONE_ENVIRONMENT=SET-ME-PLEASE
```

### Pinecone setup

You'll need to manually create an index with the following characteristics

- Environment: gcp-starter
- Index name: netec-rag
- Metric: dotproduct
- Dimensions: 1536
- Pod Type: starter

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
