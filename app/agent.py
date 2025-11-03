"""Agent with LangChain for tool routing and response generation."""
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage

from app.retriever import HybridRetriever
from app.tools.prices import PriceTool
from app.observability import metrics_collector, timer, generate_query_id
from app.models import QueryResponse, QueryMetadata, Citation

load_dotenv()


class RAGAgent:
    """RAG Agent with tool routing capabilities."""
    
    def __init__(self):
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in the environment for RAGAgent.")

        self.retriever = HybridRetriever()
        self.price_tool = PriceTool()
        self._configure_langchain_tracing()

        # Initialize LLM and LangChain agent
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=0,
            api_key=self.openai_api_key
        )

        self.tools: List[Tool] = self._create_tools()
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools for the agent."""
        
        def retrieval_tool_func(query: str) -> str:
            """Retrieve relevant documents for a query."""
            with timer() as t:
                chunks = self.retriever.retrieve(query, top_k=5)
            
            # Log retrieval
            query_id = getattr(self, '_current_query_id', 'unknown')
            metrics_collector.log_retrieval(query_id, chunks, t['elapsed_ms'])
            metrics_collector.log_tool_usage(query_id, "document_search", True, t['elapsed_ms'])
            
            # Store chunks for citation extraction
            self._last_retrieved_chunks = chunks
            
            # Format as context
            return self.retriever.format_chunks_for_context(chunks)
        
        def price_tool_func(query: str) -> str:
            """Get price information for tickers."""
            with timer() as t:
                result = self.price_tool.query_prices(query)
            
            # Log tool usage
            query_id = getattr(self, '_current_query_id', 'unknown')
            metrics_collector.log_tool_usage(query_id, "price_lookup", True, t['elapsed_ms'])
            
            return result
        
        retrieval_tool = Tool(
            name="document_search",
            func=retrieval_tool_func,
            description="""Use this tool to search through the fund letters, macro addendums, and chat logs.
            This tool should be used for questions about:
            - Investment strategy, factor tilts, methodology
            - Market observations, breadth, concentration
            - Q2 performance and attribution
            - Risk management and liquidity filters
            - Team discussions and operational notes
            Input should be the query text."""
        )
        
        price_tool = Tool(
            name="price_lookup",
            func=price_tool_func,
            description="""Use this tool to get current prices, historical prices, or compare ticker performance.
            This tool should be used for questions about:
            - Latest/recent/current prices for tickers (AAPL, MSFT, SPY, QQQ, EURUSD)
            - Price comparisons between tickers
            - Performance over time periods
            Input should be the query text mentioning ticker symbols."""
        )
        
        return [retrieval_tool, price_tool]
    
    def _create_agent(self):
        """Create the OpenAI functions agent."""
        
        # Load system prompt
        system_prompt = self._load_system_prompt()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return agent
    
    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        prompt_path = "prompts/answer_system.txt"
        
        default_prompt = """You are a helpful financial research assistant.

        Answer questions using ONLY the information provided by the tools:
        - Use 'document_search' for questions about strategy, methodology, market observations, and team discussions
        - Use 'price_lookup' for questions about ticker prices and performance

        When answering from documents:
        1. Base your answer ONLY on the retrieved context
        2. Cite sources using the format [source@chunk_index]
        3. If the information is not in the context, say so clearly

        When answering with prices:
        1. Use the exact prices and dates provided by the price tool
        2. Format numbers clearly with currency symbols
        3. Explain comparisons clearly

        Be concise and precise. Always cite your sources."""
        
        try:
            with open(prompt_path, 'r') as f:
                base_prompt = f.read().strip()
                return f"{base_prompt}\n\n{default_prompt}"
        except FileNotFoundError:
            return default_prompt
    
    def query(self, query_text: str) -> QueryResponse:
        """Process a query and return a response.
        
        Args:
            query_text: The user's question
            
        Returns:
            QueryResponse with answer, citations, and metadata
        """
        # Generate query ID and start timing
        query_id = generate_query_id()
        self._current_query_id = query_id
        self._last_retrieved_chunks = []
        
        # Log query
        metrics_collector.log_query(query_id, query_text, "rag")
        
        self._log_tracing_metric(query_id)

        with timer() as total_timer:
            try:
                # Execute agent
                result = self.agent_executor.invoke({"input": query_text})
                answer = result.get("output", "I couldn't generate an answer.")
                
                # Extract citations from the answer and retrieved chunks
                citations = self._extract_citations(answer)
                sources = list(set([c.source for c in citations]))
                
                # Determine which tools were used (from logs)
                tools_used = self._get_tools_used()
                
                # Create metadata
                metadata = QueryMetadata(
                    query_id=query_id,
                    total_latency_ms=total_timer['elapsed_ms'],
                    chunks_retrieved=len(self._last_retrieved_chunks),
                    tools_used=tools_used
                )
                
                # Log response
                metrics_collector.log_response(
                    query_id,
                    answer,
                    len(citations),
                    total_timer['elapsed_ms']
                )
                
                return QueryResponse(
                    answer=answer,
                    citations=citations,
                    sources=sources,
                    metadata=metadata
                )
                
            except Exception as e:
                print(f"Error processing query: {e}")
                return QueryResponse(
                    answer=f"Error processing query: {str(e)}",
                    citations=[],
                    sources=[],
                    metadata=QueryMetadata(
                        query_id=query_id,
                        total_latency_ms=total_timer['elapsed_ms'],
                        chunks_retrieved=0,
                        tools_used=[]
                    )
                )

    def _extract_citations(self, answer: str) -> List[Citation]:
        """Extract citations from the answer text.
        
        Looks for patterns like [source@chunk_index] in the answer.
        """
        import re
        
        citations = []
        
        # Pattern: [filename@chunk_index]
        pattern = r'\[([^\]@]+)@(\d+)\]'
        matches = re.findall(pattern, answer)
        
        for source, chunk_idx in matches:
            # Find the corresponding chunk
            chunk_idx_int = int(chunk_idx)
            matching_chunks = [
                c for c in self._last_retrieved_chunks
                if c['metadata'].get('source') == source and
                c['metadata'].get('chunk_index') == chunk_idx_int
            ]
            
            if matching_chunks:
                chunk = matching_chunks[0]
                citation = Citation(
                    doc_id=chunk['id'],
                    start_char=0,  # Could be more precise with actual text matching
                    end_char=len(chunk['text']),
                    text_snippet=chunk['text'][:200] + "...",  # First 200 chars
                    source=source
                )
                citations.append(citation)
        
        return citations
    
    def _get_tools_used(self) -> List[str]:
        """Extract which tools were used from recent metrics."""
        # This is a simplified version - in a real system you'd track this more carefully
        tools_used = []
        if self._last_retrieved_chunks:
            tools_used.append("document_search")
        # Could check metrics_collector logs for price tool usage
        return tools_used

    def _configure_langchain_tracing(self) -> None:
        """Enable LangChain/LangSmith tracing if an API key is present."""
        if self.langchain_api_key:
            os.environ.setdefault("LANGCHAIN_API_KEY", self.langchain_api_key)
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "rag-agent"))
            self.langsmith_tracing_enabled = True
        else:
            self.langsmith_tracing_enabled = False

    def _log_tracing_metric(self, query_id: str) -> None:
        """Emit a metric capturing tracing enablement for auditing."""
        metrics_collector.log_metric(
            "langchain_tracing_enabled",
            1 if self.langsmith_tracing_enabled else 0,
            query_id,
            {
                "project": os.getenv("LANGCHAIN_PROJECT", ""),
                "tracing": self.langsmith_tracing_enabled,
            }
        )
