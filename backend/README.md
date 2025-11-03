# Backend Contracts

The frontend expects a backend that exposes the following HTTP API. All endpoints are served relative to `VITE_BACKEND_URL` (e.g. `https://rag.example.com`). Streaming responses use **Server-Sent Events (SSE)**.

## Authentication

If authentication is required, expose it via standard HTTP mechanisms (e.g. `Authorization` header). The frontend forwards any values you inject into `fetch` globally (adjust `lib/api.ts` if a custom approach is required).

## Endpoints

### `POST /api/ask`
Initiates an answer generation stream for a user query.

**Request body**
```json
{
  "query": "string",
  "sessionId": "optional string"
}
```

**Response**: `text/event-stream`

Streamed events (SSE format):
- `event: token` → `data: { "content": "..." }`
- `event: citation` → `data: { "citation": Citation }`
- `event: done` → `data: { "messageId": "uuid", "usage": { "promptTokens": 123, "completionTokens": 456 } }`
- `event: error` → `data: { "message": "description" }`

`Citation` matches the type defined in `/frontend/src/types.ts` (`docId`, `docType`, `title`, optional `page`, `highlights`, etc.). Events can be interleaved; citations may arrive before or after tokens.

### `GET /api/docs`
Returns metadata for all ingested documents.

**Response body**
```json
{
  "docs": DocMeta[]
}
```

`DocMeta` is defined in `/frontend/src/types.ts` and includes the download `url` for the document.

### `GET /api/docs/:id`
Returns metadata for a single document.

**Response body**
```json
{
  "doc": DocMeta
}
```

### `GET /api/docs/:id/file`
Returns the raw document (PDF/HTML/CSV). The frontend streams the file directly into the viewers and expects the appropriate `Content-Type` headers.

### `GET /api/metrics` (optional)
Returns recent observability events for display in the metrics table.

**Response body**
```json
{
  "events": Array<{
    "time": "ISO 8601 timestamp",
    "type": "string",
    "summary": "string"
  }>
}
```

## Streaming Notes

- Use standard SSE formatting (`event:` and `data:` lines separated by a blank line).
- `token` events should contain text segments that can be concatenated in order. Avoid sending empty payloads.
- When a response finishes normally, emit a `done` event. If the stream needs to terminate early, emit an `error` event with a descriptive message.

## Citation Highlights

PDF highlights are an array of rectangles in **normalized coordinates** (values between 0 and 1). The frontend converts them to pixel coordinates using the rendered PDF viewport size. HTML/CSV highlights can provide `text` snippets to wrap with a `.highlight` mark.

---

The backend implementation is out of scope for this repository. Use this document as a contract when wiring the UI to a real service.
