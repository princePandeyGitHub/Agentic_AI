# Agentic AI System

## High Level Flow

### ingest the hr_data and incident_data to the chromdb first
- load the embedding model
- create chromadb client
- get / create the collections
- load the json data ( we have two types of data hr(which includes information about employees an d as well as policies) and incident(which includes information about different incidents as well as trouble shooting guides))
- create four different types of data to be stored in the chromadb
- convert each type of data to text and make its embedding and stores to chromadb

### how to search in these documents(tests)
- similar steps load the model, chromadb client and get the collections
- prepare your query and then do its embedding also and keep consistent while embedding
- now we can simply search with this embedding with built-in function and n_results return the top k documents we want
- it can return these things
```bash
{
  "documents": [[doc1, doc2, doc3]],
  "metadatas": [[meta1, meta2, meta3]],
  "distances": [[d1, d2, d3]],
  "ids": [[id1, id2, id3]]
}

```
- we usually use documents to perform operations

## working of the agent.py
- steps being repeated
- we loads everything similarly and also setups our llm model
- and then ask the llm with three things 
- detailed prompt(that what llm have to do)
- user query(what the user wants)
- context(context is nothing but the top k documents we can get through the embedding model from chromadb)