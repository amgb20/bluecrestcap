# Front-Office RAG Agent Excercise

## Instructions

Build a small RAG service that:
- Ingests the provided **HTML letter**, **PDF addendum**, and **chat CSV** (plus any additional data you might wish to include)
- Answers questions with **grounded, cited** responses
- Calls a **price tool** for price questions (from `prices_stub/prices.json`)
- Emits **metrics** so we can evaluate quality & performance

## Notes

- Please feel free to change any aspect of this repository, apart from the data provided.
- The files and project structure provided simply serve as an example scaffolding, or starting point for you to work with.

## Get up & running

```bash
cp .env.example .env                 # set any keys you need
docker compose -f docker/docker-compose.yml up -d qdrant   # optional if you use Qdrant
make ingest
make run
make test
make eval                            # run the simple eval harness
```
