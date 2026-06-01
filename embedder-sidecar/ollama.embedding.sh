curl http://localhost:11434/v1/embeddings \                                                                                               [19:53:02]
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding:0.6b",
    "input": "The sky is blue because of Rayleigh scattering."
  }'

curl http://localhost:11434/api/tags | jq


