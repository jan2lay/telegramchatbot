from milvus import default_server

with default_server:
  # Milvus Lite has already started, use default_server here.
  connections.connect(host='127.0.0.1', port=default_server.listen_port)