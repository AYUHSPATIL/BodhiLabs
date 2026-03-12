# import chromadb

# # Connect to your local vector store
# client = chromadb.PersistentClient(path="./vectorstore/chroma")

# # Get the collection
# collection = client.get_or_create_collection("module_341_urinary_catheterization")

# # Check how many items
# print(f"Total items: {collection.count()}")

# # Peek at some data
# results = collection.peek(limit=5)
# print(results)

# # Query example
# query_results = collection.query(
#     query_texts=["urinary catheterization procedure"],
#     n_results=3
# )
# print(query_results)

