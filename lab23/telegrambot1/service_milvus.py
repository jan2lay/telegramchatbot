from milvus import default_server
from pymilvus import connections, utility

# (OPTIONAL) Set if you want store all related data to specific location
# Default location:
#   %APPDATA%/milvus-io/milvus-server on windows
#   ~/.milvus-io/milvus-server on linux
# default_server.set_base_dir('milvus_data')

# (OPTIONAL) if you want cleanup previous data
# default_server.cleanup()

# Start your milvus server
default_server.start()

# Now you could connect with localhost and the given port
# Port is defined by default_server.listen_port
connections.connect(host='127.0.0.1', port=default_server.listen_port)

# Check if the server is ready.
print(utility.get_server_version())

# Stop your milvus server
default_server.stop()