import json
import requests
import streamlit as st
from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata, RetrieverTool

GITHUB_TOKEN = st.secrets.github_token
DEVELOPER_USERNAME = 'glmaljkovich'


def query_github() -> str:
    """
    Returns the github repos glmaljkovich has contributed to in JSON format
    """
    # The GraphQL query
    query = """
    {
        user(login: "{{username}}") {
            repositories(
                first: 10,
                isArchived: false,
                affiliations: [OWNER, COLLABORATOR],
                isFork: false,
                ownerAffiliations: [OWNER, COLLABORATOR],
                orderBy: {field: STARGAZERS, direction: DESC}
            ) {
                nodes {
                    nameWithOwner
                  	description
                  	languages(first: 3) {
                      nodes {
                        name
                      }
                    }
                    stargazers {
                        totalCount
                    }
                }
            }
        }
    }
    """.replace("{{username}}", DEVELOPER_USERNAME)

    print(f"using query {query}")
    print(f"using token: {GITHUB_TOKEN}")

    # Headers for the request
    headers = {
        'Authorization': f'Bearer {GITHUB_TOKEN}',
        'Content-Type': 'application/json',
    }

    def get_popular_repos(query):
        """
        Fetches repositories a user has contributed to and sorts them by the number of stars.
        """
        response = requests.post('https://api.github.com/graphql', headers=headers, json={'query': query})
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Query failed to run with a {response.status_code}.")
        # Fetch and process the data
    data = get_popular_repos(query)
    repositories = data['data']['user']['repositories']['nodes']
    return json.dumps(repositories)

def github_tool(): 
    return FunctionTool.from_defaults(
        fn=query_github,
        name="github",
        description="Retrieve Gabriel's Github repos and contributions in JSON format"
    )

def resume_tool(index: VectorStoreIndex):
    query_engine = index.as_query_engine(similarity_top_k=3)
    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="resume",
            description=(
                "Query facts from Gabriel's work history from his resume"
            ),
        ),
    )

def resume_summary_tool(index: SummaryIndex):
    query_engine = index.as_retriever()
    return RetrieverTool(
        retriever=query_engine,
        metadata=ToolMetadata(
            name="resume_summary",
            description=(
                "Query Gabriel's work history from his resume and summarize information. Useful for enumerating."
                "Reply as if you were Gabriel, you can use markdown"
            ),
        ),
    )