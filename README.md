# Structured-hierarchical-Retrieval---RAG
https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval.ipynb
Llama index RAG Solution - represents documents as concise meta-data dictionaries in a vector databsae

Structured Hierarchical Retrieval
Open In Colab

Doing RAG well over multiple documents is hard. A general framework is given a user query, first select the relevant documents before selecting the content inside.

But selecting the documents can be tough - how can we dynamically select documents based on different properties depending on the user query?

In this notebook we show you our multi-document RAG architecture:

Represent each document as a concise metadata dictionary containing different properties: an extracted summary along with structured metadata.
Store this metadata dictionary as filters within a vector database.
Given a user query, first do auto-retrieval - infer the relevant semantic query and the set of filters to query this data (effectively combining text-to-SQL and semantic search).
!pip install llama-index
Setup and Download Data
In this section, we'll load in LlamaIndex Github issues.

import nest_asyncio

nest_asyncio.apply()
import os

os.environ["GITHUB_TOKEN"] = ""
import os

from llama_hub.github_repo_issues import (
    GitHubRepositoryIssuesReader,
    GitHubIssuesClient,
)

github_client = GitHubIssuesClient()
loader = GitHubRepositoryIssuesReader(
    github_client,
    owner="run-llama",
    repo="llama_index",
    verbose=True,
)

orig_docs = loader.load_data()

limit = 100

docs = []
for idx, doc in enumerate(orig_docs):
    doc.metadata["index_id"] = doc.id_
    if idx >= limit:
        break
    docs.append(doc)
Found 100 issues in the repo page 1
Resulted in 100 documents
Found 100 issues in the repo page 2
Resulted in 200 documents
Found 100 issues in the repo page 3
Resulted in 300 documents
Found 9 issues in the repo page 4
Resulted in 309 documents
No more issues found, stopping
from copy import deepcopy
import asyncio
from tqdm.asyncio import tqdm_asyncio
from llama_index import SummaryIndex, Document, ServiceContext
from llama_index.llms import OpenAI
from llama_index.async_utils import run_jobs


async def aprocess_doc(doc, include_summary: bool = True):
    """Process doc."""
    print(f"Processing {doc.id_}")
    metadata = doc.metadata

    date_tokens = metadata["created_at"].split("T")[0].split("-")
    year = int(date_tokens[0])
    month = int(date_tokens[1])
    day = int(date_tokens[2])

    assignee = (
        "" if "assignee" not in doc.metadata else doc.metadata["assignee"]
    )
    size = ""
    if len(doc.metadata["labels"]) > 0:
        size_arr = [l for l in doc.metadata["labels"] if "size:" in l]
        size = size_arr[0].split(":")[1] if len(size_arr) > 0 else ""
    new_metadata = {
        "state": metadata["state"],
        "year": year,
        "month": month,
        "day": day,
        "assignee": assignee,
        "size": size,
        "index_id": doc.id_,
    }

    # now extract out summary
    summary_index = SummaryIndex.from_documents([doc])
    query_str = "Give a one-sentence concise summary of this issue."
    query_engine = summary_index.as_query_engine(
        service_context=ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo")
        )
    )
    summary_txt = str(query_engine.query(query_str))

    new_doc = Document(text=summary_txt, metadata=new_metadata)
    return new_doc


async def aprocess_docs(docs):
    """Process metadata on docs."""

    new_docs = []
    tasks = []
    for doc in docs:
        task = aprocess_doc(doc)
        tasks.append(task)

    new_docs = await run_jobs(tasks, show_progress=True, workers=5)

    # new_docs = await tqdm_asyncio.gather(*tasks)

    return new_docs
new_docs = await aprocess_docs(docs)
  0%|                                                                                                                                  | 0/100 [00:00<?, ?it/s]
Processing 9620
Processing 9312
Processing 9435
Processing 9576
Processing 9219
Processing 9571
Processing 9383
Processing 9425
Processing 9405
Processing 9624
Processing 9419
Processing 9546
Processing 9373
Processing 9565
Processing 9408
Processing 9372
Processing 9560
Processing 9415
Processing 9414
Processing 9097
Processing 9492
Processing 9358
Processing 9385
Processing 9269
Processing 9380
Processing 8802
Processing 9352
Processing 9525
Processing 9368
Processing 9543
Processing 8893
Processing 8551
Processing 9470
Processing 9342
Processing 9518
Processing 9343
Processing 9488
Processing 9338
Processing 9337
Processing 9335
Processing 9623
Processing 9314
Processing 8536
Processing 9510
Processing 9523
Processing 9416
Processing 9522
Processing 9520
Processing 7244
Processing 9519
Processing 9602
Processing 9507
Processing 9605
Processing 9491
Processing 9490
Processing 9611
Processing 9353
Processing 3258
Processing 9575
Processing 9348
Processing 7299
Processing 9625
Processing 9483
Processing 9630
Processing 9481
Processing 9627
Processing 9469
Processing 9626
Processing 9477
Processing 9164
Processing 9450
Processing 9398
Processing 9613
Processing 9459
Processing 9612
Processing 9394
Processing 8832
Processing 9439
Processing 9421
Processing 9609
Processing 9413
Processing 9618
Processing 9509
Processing 9574
Processing 9339
Processing 9603
Processing 9604
Processing 9427
Processing 7457
Processing 9417
Processing 9583
Processing 9581
Processing 9426
Processing 7720
Processing 9475
Processing 9431
Processing 9471
Processing 9472
Processing 9607
Processing 9531
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [01:58<00:00,  1.18s/it]
new_docs[5].metadata
{'state': 'open',
 'year': 2023,
 'month': 12,
 'day': 19,
 'assignee': '',
 'size': '',
 'index_id': '9611'}
Load Data into Vector Store
We load both the summarized metadata as well as the original docs into the vector database.

Summarized Metadata: This goes into the LlamaIndex_auto collection.
Original Docs: This goes into the LlamaIndex_AutoDoc collection.
By storing both the summarized metadata as well as the original documents, we can execute our structured, hierarchical retrieval strategies.

We load into a vector database that supports auto-retrieval.

Load Summarized Metadata
This goes into LlamaIndex_auto

from llama_index.vector_stores import WeaviateVectorStore
from llama_index.storage import StorageContext
from llama_index import VectorStoreIndex
import weaviate

# cloud
auth_config = weaviate.AuthApiKey(api_key="")
client = weaviate.Client(
    "https://<weaviate-cluster>.weaviate.network",
    auth_client_secret=auth_config,
)

class_name = "LlamaIndex_auto"
# optional: delete schema
client.schema.delete_class(class_name)
vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name=class_name
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# Since "new_docs" are concise summaries, we can directly feed them as nodes into VectorStoreIndex
index = VectorStoreIndex(new_docs, storage_context=storage_context)
Load Original Docs
This goes into LlamaIndex_AutoDoc.

In later sections we'll create N different query engines from this collection, each query engine pointing to a specific doc (effective creating doc-specific RAG pipelines).

docs[0].metadata
{'state': 'open',
 'created_at': '2023-12-19T22:10:49Z',
 'url': 'https://api.github.com/repos/run-llama/llama_index/issues/9624',
 'source': 'https://github.com/run-llama/llama_index/pull/9624',
 'labels': ['size:L'],
 'index_id': '9624'}
# optional: delete schema
doc_class_name = "LlamaIndex_AutoDoc"
client.schema.delete_class(doc_class_name)
# construct separate Weaviate Index with original docs. Define a separate query engine with query engine mapping to each doc id.
vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name=doc_class_name
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

doc_index = VectorStoreIndex.from_documents(
    docs, storage_context=storage_context
)
Setup Auto-Retriever
In this section we setup our auto-retriever. There's a few steps that we need to perform.

Define the Schema: Define the vector db schema (e.g. the metadata fields). This will be put into the LLM input prompt when it's deciding what metadata filters to infer.
Instantiate the VectorIndexAutoRetriever class: This creates a retriever on top of our summarized metadata index, and takes in the defined schema as input.
Define a wrapper retriever: This allows us to postprocess each node into an IndexNode, with an index id linking back source document. This will allow us to do recursive retrieval in the next section (which depends on IndexNode objects linking to downstream retrievers/query engines/other Nodes). NOTE: We are working on improving this abstraction.
1. Define the Schema
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo


vector_store_info = VectorStoreInfo(
    content_info="Github Issues",
    metadata_info=[
        MetadataInfo(
            name="state",
            description="Whether the issue is `open` or `closed`",
            type="string",
        ),
        MetadataInfo(
            name="year",
            description="The year issue was created",
            type="integer",
        ),
        MetadataInfo(
            name="month",
            description="The month issue was created",
            type="integer",
        ),
        MetadataInfo(
            name="day",
            description="The day issue was created",
            type="integer",
        ),
        MetadataInfo(
            name="assignee",
            description="The assignee of the ticket",
            type="string",
        ),
        MetadataInfo(
            name="size",
            description="How big the issue is (XS, S, M, L, XL, XXL)",
            type="string",
        ),
    ],
)
2. Instantiate VectorIndexAutoRetriever
from llama_index.retrievers import VectorIndexAutoRetriever

retriever = VectorIndexAutoRetriever(
    index,
    vector_store_info=vector_store_info,
    similarity_top_k=2,
    empty_query_top_k=10,  # if only metadata filters are specified, this is the limit
    verbose=True,
)
Try It Out
We can try out our autoretriever on its own.

nodes = retriever.retrieve("Tell me about some issues on 12/11")
print(f"Number retrieved: {len(nodes)}")
print(nodes[0].metadata)
Using query str: 
Using filters: {'month': 12, 'day': 11}
Number retrieved: 6
{'state': 'open', 'year': 2023, 'month': 12, 'day': 11, 'assignee': '', 'size': '', 'index_id': '9435'}
3. Define a Wrapper Retriever
from llama_index.retrievers import BaseRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import IndexNode, NodeWithScore


class IndexAutoRetriever(BaseRetriever):
    """Index auto-retriever."""

    def __init__(self, retriever: VectorIndexAutoRetriever):
        """Init params."""
        self.retriever = retriever

    def _retrieve(self, query_bundle: QueryBundle):
        """Convert nodes to index node."""
        retrieved_nodes = self.retriever.retrieve(query_bundle)
        new_retrieved_nodes = []
        for retrieved_node in retrieved_nodes:
            index_id = retrieved_node.metadata["index_id"]
            index_node = IndexNode.from_text_node(
                retrieved_node.node, index_id=index_id
            )
            new_retrieved_nodes.append(
                NodeWithScore(node=index_node, score=retrieved_node.score)
            )
        return new_retrieved_nodes
index_retriever = IndexAutoRetriever(retriever=retriever)
Setup Recursive Retriever
Now we setup a recursive retriever over our data. A recursive retriever links each node of one retriever to another retriever, query engine, or Node.

In this setup, we link each summarized metadata node to a retriever corresponding to a RAG pipeline over the corresponding document.

We set this up through the following:

Define one retriever per document: Put this in a dictionary
Define our recursive retriever: Add the root retriever (the summarized metadata retriever), and add the other document-specific retrievers in the arguments.
1. Define Per-Document Retriever
from llama_index.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
retriever_dict = {}
query_engine_dict = {}
for doc in docs:
    index_id = doc.metadata["index_id"]
    # filter for the specific doc id
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="index_id", operator=FilterOperator.EQ, value=index_id
            ),
        ]
    )
    retriever = doc_index.as_retriever(filters=filters)
    query_engine = doc_index.as_query_engine(filters=filters)

    retriever_dict[index_id] = retriever
    query_engine_dict[index_id] = query_engine
2. Define Recursive Retriever
We can now define our recursive retriever, which will first query the summaries and then retrieve the underlying docs.

from llama_index.retrievers import RecursiveRetriever

# note: can pass `agents` dict as `query_engine_dict` since every agent can be used as a query engine
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": index_retriever, **retriever_dict},
    # query_engine_dict=query_engine_dict,
    verbose=True,
)
Try It Out
Now we can start retrieving relevant context over Github Issues!

To complete the RAG pipeline setup we'll combine our recursive retriever with our RetrieverQueryEngine to generate a response in addition to the retrieved nodes.

Try Out Retrieval
nodes = recursive_retriever.retrieve("Tell me about some issues on 12/11")
If you ran the above, you should've gotten a long output in the logs.

The result is the source chunks in the relevant docs.

Let's look at the date attached to the source chunk (was present in the original metadata).

print(f"Number of source nodes: {len(nodes)}")
nodes[0].node.metadata
Number of source nodes: 6
{'state': 'open',
 'created_at': '2023-12-11T14:03:19Z',
 'url': 'https://api.github.com/repos/run-llama/llama_index/issues/9435',
 'source': 'https://github.com/run-llama/llama_index/issues/9435',
 'labels': ['bug', 'triage'],
 'index_id': '9435'}
Plug into RetrieverQueryEngine
We plug into RetrieverQueryEngine to synthesize a result.

from llama_index.query_engine import RetrieverQueryEngine
from llama_index.llms import OpenAI
from llama_index import ServiceContext


llm = OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

query_engine = RetrieverQueryEngine.from_args(recursive_retriever, llm=llm)
response = query_engine.query("Tell me about some issues on 12/11")
print(str(response))
There were several issues created on 12/11. One of them is a bug where the metadata filter is not working correctly with Elastic search indexing. Another bug involves an error loading the 'punkt' module in the NLTK library. There are also a couple of feature requests, one for adding Postgres BM25 support and another for making llama-index compatible with models finetuned and hosted on modal.com. Additionally, there is a question about using the Slack Loader with large Slack channels.
response = query_engine.query(
    "Tell me about some open issues related to agents"
)
Retrieving with query id None: Tell me about some open issues related to agents
Using query str: agents
Using filters: {'state': 'open'}
Retrieved node with id, entering: 9472
Retrieving with query id 9472: Tell me about some open issues related to agents
Retrieving text node: [Feature Request]: Add stop words to ReAct agent
### Feature Description

The ReAct agent does not use any stop words and the current API does not allow these to be passed to the LLM API.
When using the ReAct agent chat abstraction the LLM often will generate an entire conversation before this output is collected by llama-index and then trimmed to the first `Thought:`, `Action:` set.

This is very, very slow for some models.

A better approach would be to use any available stop word setting in the APIs llama-index calls, or to instead use a streaming approach and implement stop words when possible this way.

Additionally stop words should be plumbed up to the chat, query, etc API. This could probably be its own issue.

### Reason

`ReActOutputParser` selects the first `Thought:`, `Action:` set to act on. This hides that the LLM is doing a lot of useless work.

`ReActAgent` should probably inject a stop word. If you build a chat or query from this the LLM will do a lot of work before its output is truncated to the first `Thought:`, `Action:` block.

### Value of Feature

LLM usage is expensive and especially slow when working locally. Currently with a variety of models, the ReAct agent is very inefficient because it generates large outputs containing many `Thought:`, `Action:` blocks and truncates to the first one. It should just avoid generating these large blocks with a stop word or by using streaming if available and stopping after the first block.

This would reduce cost and significantly increase speed for local inference.
Retrieved node with id, entering: 9565
Retrieving with query id 9565: Tell me about some open issues related to agents
Retrieving text node: [Question]: Connecting to Mixtral 8x7b Chat on together.ai
### Question Validation

- [X] I have searched both the documentation and discord for an answer.

### Question

I am trying to connect to mistral chat model on together.ai

model is defined as OpenAILike
llm = OpenAILike(
    model="DiscoResearch/DiscoLM-mixtral-8x7b-v2",
    api_base="https://api.together.xyz/v1",
    api_key="<secret key>",
    temperatue=0.1
)

but I am not getting any responses as I suspect that model is expecting specific prompt template.
Anyone managed to make it work, quick sample would be appreciated ?
print(str(response))
There are two open issues related to agents. The first issue is a feature request to add stop words to the ReAct agent. The issue describes that the ReAct agent does not currently use any stop words, which results in slow performance for some models. The request suggests using stop words in the APIs or implementing a streaming approach to improve efficiency. The second issue is a question about connecting to the Mistral 8x7b chat model on together.ai. The user is seeking assistance in making the model work and is looking for a sample prompt template.
response = query_engine.query(
    "Tell me about some size S issues related to our llm integrations"
)
Retrieving with query id None: Tell me about some size S issues related to our llm integrations
Using query str: llm integrations
Using filters: {'size': 'S'}
Retrieved node with id, entering: 9421
Retrieving with query id 9421: Tell me about some size S issues related to our llm integrations
Retrieving text node: Fix cleanup process in  _delete_node of document_summary
# Description

Cannot remove doc from DocumentSummaryIndex by delete_ref_doc(...)

As designed, the user can remove the nodes from the document summary index according to "doc_id" by delete_ref_doc(...) which will call delete_nodes(...) from BaseIndex to do the work. 
<img width="600" alt="截屏2023-12-10 16 12 57" src="https://github.com/run-llama/llama_index/assets/114048/d1b804fd-9830-4dce-985f-ac39acffcb4d">

However, it passes the related node_ids instead of doc_id itself. 
<img width="692" alt="截屏2023-12-10 16 10 15" src="https://github.com/run-llama/llama_index/assets/114048/68e89cb2-d63d-40cc-adee-1dcec1ee443a">

Fixes # (issue)

## Type of Change

Please delete options that are not relevant.

- [x] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

# How Has This Been Tested?

Please describe the tests that you ran to verify your changes. Provide instructions so we can reproduce. Please also list any relevant details for your test configuration

- [ ] Added new unit/integration tests
- [ ] Added new notebook (that tests end-to-end)
- [x] I stared at the code and made sure it makes sense

# Suggested Checklist:

- [x] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] I have added Google Colab support for the newly added notebooks.
- [x] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
- [ ] I ran `make format; make lint` to appease the lint gods
Retrieved node with id, entering: 9372
Retrieving with query id 9372: Tell me about some size S issues related to our llm integrations
Retrieving text node: Proposal : Update the evaluation correctness function to be more robust.
# Description

Proposal : Update the evaluation correctness function to be more robust.

When using RagEvaluatorPack on a large dataset, sometime GPT3/4 will return a malformated answer, raising an error in correctness.py and interumpting the benchmark (that could be costly).

Such case emerge when the LLM :
 - prefix the answer with ```\n```
 - do not answer correctly such as: ```I'm not sure how to evaluate this case so I will say 3.0```
 - Something in the content made the LLM go off-road
 
To make the parsing more robust, I change the prompt to output the score in the form ```[SCORE:4.2]``` instead of only a number.

I then use a regexp to retrieve the score instead of assuming first line.

I use a regexp to remove the ```[SCORE:2.3]``` pattern from the llm answer to get a reasoning, without relying on line marker.

## Type of Change

Please delete options that are not relevant.

- [x] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

# How Has This Been Tested?

Please describe the tests that you ran to verify your changes. Provide instructions so we can reproduce. Please also list any relevant details for your test configuration

- [ ] Added new unit/integration tests
- [ ] Added new notebook (that tests end-to-end)
- [x] I stared at the code and made sure it makes sense

# Suggested Checklist:

- [x] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] I have added Google Colab support for the newly added notebooks.
- [x] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I ran `make format; make lint` to appease the lint gods
print(str(response))
There are two size S issues related to the llm integrations. The first issue is about fixing the cleanup process in the _delete_node of document_summary. The second issue is a proposal to update the evaluation correctness function to be more robust. Both issues are currently open and have not been resolved yet.
Concluding Thoughts
This shows you how to create a structured retrieval layer over your document summaries, allowing you to dynamically pull in the relevant documents based on the user query.

You may notice similarities between this and our multi-document agents. Both architectures are aimed for powerful multi-document retrieval.

The goal of this notebook is to show you how to apply structured querying in a multi-document setting. You can actually apply this auto-retrieval algorithm to our multi-agent setup too. The multi-agent setup is primarily focused on adding agentic reasoning across documents and per documents, alloinwg multi-part queries using chain-of-thought.
