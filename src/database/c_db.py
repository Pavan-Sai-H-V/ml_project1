import chromadb
chromaClient=chromadb.Client()

collection=chromaClient.create_collection(name='collection1')


collection.add(
    ids=['1','2'],
    documents=['this is sample document','this is sample document 2']
)


results=collection.query(
    query_texts=['this is a question about hawaii'],
    n_results=2
)

print(results)