from qdrant_client import QdrantClient

client = QdrantClient(
    url="https://YOUR-CLOUD-CLUSTER.qdrant.tech",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.vhcGFJddkKGCp7nBm7GsNLi_c7L7QsFTaXw9_AwxsLE"
)
print(client.get_collections())
