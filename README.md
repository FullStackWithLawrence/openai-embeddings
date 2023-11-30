# Netec Large Language Models

A Python [LangChain](https://www.langchain.com/) - [Pinecone](https://docs.pinecone.io/docs/python-client) proof of concept LLM to manage sales support inquiries on the Netec course catalogue.

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
