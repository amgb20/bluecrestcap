"""Evaluation harness for the RAG agent."""
import json
import os
import time
import requests
from pathlib import Path
from typing import List, Dict, Any
from statistics import mean, median
import sys

from fastapi.testclient import TestClient

from app.main import app as fastapi_app


class EvaluationHarness:
    """Run evaluation queries and collect metrics."""
    
    def __init__(self, api_url: str = "http://localhost:8000", use_inprocess: bool = True):
        self.api_url = api_url.rstrip("/")
        self.use_inprocess = use_inprocess
        self.results: List[Dict[str, Any]] = []
        self._client_ctx = None
        if use_inprocess:
            self._client_ctx = TestClient(fastapi_app)
            self.client = self._client_ctx.__enter__()
        else:
            self.client = requests.Session()
    
    def load_queries(self, queries_file: str = "eval/queries.jsonl") -> List[Dict[str, str]]:
        """Load queries from JSONL file."""
        queries = []
        with open(queries_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(json.loads(line))
        return queries
    
    def run_query(self, query: str) -> Dict[str, Any]:
        """Run a single query against the API."""
        start_time = time.time()
        
        try:
            if self.use_inprocess:
                response = self.client.post(
                    "/query",
                    json={"query": query},
                    timeout=30
                )
            else:
                response = self.client.post(
                    f"{self.api_url}/query",
                    json={"query": query},
                    timeout=30
                )
            response.raise_for_status()
            result = response.json()
            
            latency = (time.time() - start_time) * 1000  # ms
            
            return {
                "query": query,
                "success": True,
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "sources": result.get("sources", []),
                "metadata": result.get("metadata", {}),
                "latency_ms": latency,
                "error": None
            }
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                "query": query,
                "success": False,
                "answer": "",
                "citations": [],
                "sources": [],
                "metadata": {},
                "latency_ms": latency,
                "error": str(e)
            }
    
    def run_evaluation(self, queries: List[Dict[str, str]]):
        """Run all evaluation queries."""
        print(f"\n{'='*80}")
        print(f"Running evaluation with {len(queries)} queries...")
        print(f"{'='*80}\n")
        
        for idx, query_data in enumerate(queries, 1):
            query = query_data.get("q", "")
            print(f"\n[{idx}/{len(queries)}] Query: {query}")
            
            result = self.run_query(query)
            self.results.append(result)
            
            if result["success"]:
                print(f"✓ Success ({result['latency_ms']:.0f}ms)")
                print(f"  Answer: {result['answer'][:150]}{'...' if len(result['answer']) > 150 else ''}")
                print(f"  Citations: {len(result['citations'])}, Sources: {result['sources']}")
            else:
                print(f"✗ Failed: {result['error']}")
            
            # Brief pause between queries
            time.sleep(0.5)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate evaluation report with metrics."""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        latencies = [r["latency_ms"] for r in successful]
        citation_counts = [len(r["citations"]) for r in successful]
        
        # Calculate percentiles
        def percentile(data, p):
            if not data:
                return 0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p
            f = int(k)
            c = k - f
            if f + 1 < len(sorted_data):
                return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
            return sorted_data[f]
        
        report = {
            "summary": {
                "total_queries": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(self.results) * 100 if self.results else 0
            },
            "latency": {
                "mean_ms": mean(latencies) if latencies else 0,
                "median_ms": median(latencies) if latencies else 0,
                "p50_ms": percentile(latencies, 0.50) if latencies else 0,
                "p95_ms": percentile(latencies, 0.95) if latencies else 0,
                "p99_ms": percentile(latencies, 0.99) if latencies else 0,
                "min_ms": min(latencies) if latencies else 0,
                "max_ms": max(latencies) if latencies else 0
            },
            "citations": {
                "mean_per_query": mean(citation_counts) if citation_counts else 0,
                "queries_with_citations": sum(1 for c in citation_counts if c > 0),
                "citation_coverage_pct": sum(1 for c in citation_counts if c > 0) / len(citation_counts) * 100 if citation_counts else 0
            },
            "sources": {
                "unique_sources": len(set(s for r in successful for s in r["sources"])),
                "avg_sources_per_query": mean([len(r["sources"]) for r in successful]) if successful else 0
            }
        }
        
        # Tool usage (if available in metadata)
        tools_used = {}
        for result in successful:
            for tool in result.get("metadata", {}).get("tools_used", []):
                tools_used[tool] = tools_used.get(tool, 0) + 1
        
        if tools_used:
            report["tool_usage"] = tools_used
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted evaluation report."""
        print(f"\n{'='*80}")
        print("EVALUATION REPORT")
        print(f"{'='*80}\n")
        
        # Summary
        summary = report.get("summary", {})
        print("Summary:")
        print(f"  Total queries: {summary.get('total_queries', 0)}")
        print(f"  Successful: {summary.get('successful', 0)}")
        print(f"  Failed: {summary.get('failed', 0)}")
        print(f"  Success rate: {summary.get('success_rate', 0):.1f}%")
        
        # Latency
        latency = report.get("latency", {})
        print(f"\nLatency (milliseconds):")
        print(f"  Mean: {latency.get('mean_ms', 0):.0f}ms")
        print(f"  Median: {latency.get('median_ms', 0):.0f}ms")
        print(f"  P50: {latency.get('p50_ms', 0):.0f}ms")
        print(f"  P95: {latency.get('p95_ms', 0):.0f}ms")
        print(f"  P99: {latency.get('p99_ms', 0):.0f}ms")
        print(f"  Min: {latency.get('min_ms', 0):.0f}ms")
        print(f"  Max: {latency.get('max_ms', 0):.0f}ms")
        
        # Citations
        citations = report.get("citations", {})
        print(f"\nCitations:")
        print(f"  Mean per query: {citations.get('mean_per_query', 0):.1f}")
        print(f"  Queries with citations: {citations.get('queries_with_citations', 0)}")
        print(f"  Citation coverage: {citations.get('citation_coverage_pct', 0):.1f}%")
        
        # Sources
        sources = report.get("sources", {})
        print(f"\nSources:")
        print(f"  Unique sources used: {sources.get('unique_sources', 0)}")
        print(f"  Avg sources per query: {sources.get('avg_sources_per_query', 0):.1f}")
        
        # Tool usage
        if "tool_usage" in report:
            print(f"\nTool Usage:")
            for tool, count in report["tool_usage"].items():
                print(f"  {tool}: {count}")
        
        print(f"\n{'='*80}\n")
    
    def save_results(self, output_file: str = "eval/results.json"):
        """Save detailed results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                "results": self.results,
                "report": self.generate_report()
            }, f, indent=2)
        
        print(f"Results saved to {output_file}")

    def close(self):
        """Dispose of any underlying HTTP resources."""
        if self.use_inprocess and self._client_ctx is not None:
            self._client_ctx.__exit__(None, None, None)
            self._client_ctx = None
        elif hasattr(self.client, "close"):
            self.client.close()


def check_api_health(api_url: str) -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Main evaluation entry point."""
    api_url = os.getenv("EVAL_API_URL", "http://localhost:8000")
    use_remote = os.getenv("EVAL_USE_REMOTE", "false").lower() == "true"

    if use_remote:
        print("Checking API health...")
        if not check_api_health(api_url):
            print(f"✗ API is not reachable at {api_url}")
            print("Please start the API server first with: make run")
            sys.exit(1)
        print(f"✓ API is healthy at {api_url}\n")
    else:
        print("Running evaluation against local in-process API instance\n")
    
    harness = EvaluationHarness(api_url, use_inprocess=not use_remote)
    
    try:
        queries = harness.load_queries()
        harness.run_evaluation(queries)
        
        # Generate and print report
        report = harness.generate_report()
        harness.print_report(report)
        
        # Save results
        harness.save_results()
        
        # Exit with appropriate code (only fail if there were no successes)
        success_rate = report.get("summary", {}).get("success_rate", 0)
        if success_rate == 0:
            sys.exit(1)
        
    except FileNotFoundError as e:
        print("✗ Error: Could not find queries file")
        print(f"  {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        harness.close()


if __name__ == "__main__":
    main()
