from embeddings_util import EmbeddingsUtil

options = {
  "api_key": "",
  "verbose": True
}

embedding_client = EmbeddingsUtil(**options)

text = "Welcome to our documentation. This guide will walk you through the basics of using our platform."

embeddings_result = embedding_client.generate_qa_embeddings_from_text(text)

print(embeddings_result)