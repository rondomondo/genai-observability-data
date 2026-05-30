<!--- Hugo front matter used to generate the website version of this page:
cascade:
  body_class: otel-docs-spec
  github_repo: &repo https://github.com/open-telemetry/semantic-conventions-genai
  github_subdir: docs
  path_base_for_github_subdir: tmp/semconv-genai/docs/
  github_project_repo: *repo
linkTitle: GenAI semantic conventions
--->

# OpenTelemetry GenAI semantic conventions

Semantic conventions for generative AI (LLM, agent, embeddings, retrieval) and
Model Context Protocol (MCP) operations, split out from the main
[open-telemetry/semantic-conventions](https://github.com/open-telemetry/semantic-conventions)
repository.

## Contents

* **[Generative AI](gen-ai/README.md)**: prose and signal definitions for GenAI
  client inference, agents, tool execution, evaluation, and MCP.
* **[Registry](registry/README.md)**: auto-generated reference for attribute
  namespaces defined in this repository (`gen_ai.*`, `mcp.*`, `openai.*`).

Shared attributes, prose, and conventions not specific to GenAI continue to
live in the upstream
[open-telemetry/semantic-conventions](https://github.com/open-telemetry/semantic-conventions)
repository; docs generated here link into that repo at a pinned upstream
version when they reference non-local content.
