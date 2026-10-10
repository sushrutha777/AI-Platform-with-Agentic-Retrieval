"""Rule-based, zero-LLM tool router."""

import re
from typing import List
from pydantic import BaseModel
from app.core.logging import logger

class RoutingDecision(BaseModel):
    tools: List[str]
    needs_llm: bool
    intent: str

class AgentRouter:
    """Fast, heuristic-based router to select tools without LLM latency."""

    @staticmethod
    def route(question: str) -> RoutingDecision:
        q_lower = re.sub(r"\s+", " ", question.strip().lower())

        # 1. Direct Intents (No retrieval needed)
        greetings = [
            "hi", "hello", "hey", "good morning", "good evening", 
            "good afternoon", "greetings", "howdy", "hola", "yo"
        ]
        farewells = ["bye", "goodbye", "see you", "cya", "farewell", "take care"]
        casual = [
            "how are you", "who are you", "what are you", "thank you", 
            "thanks", "good job", "who made you", "what can you do", "help"
        ]

        # Keep greeting-only messages on the zero-latency path, but do not
        # discard a real question after a salutation such as
        # "Hi, what is the return policy?".
        greeting_prefix = sorted(greetings, key=len, reverse=True)
        for greeting in greeting_prefix:
            if q_lower == greeting:
                return RoutingDecision(tools=[], needs_llm=True, intent="greeting")
            match = re.match(rf"^{re.escape(greeting)}(?:[\s,!.?;:-]+)(.+)$", q_lower)
            if match:
                remainder = match.group(1).strip(" ,.!?;:-")
                if remainder and remainder not in {"there", "everyone", "folks"}:
                    q_lower = remainder
                else:
                    return RoutingDecision(tools=[], needs_llm=True, intent="greeting")
                break

        if any(q_lower == f for f in farewells):
            return RoutingDecision(tools=[], needs_llm=True, intent="farewell")

        # Courtesy phrases can also precede a substantive question.
        for courtesy in ["thanks", "thank you"]:
            if q_lower == courtesy:
                return RoutingDecision(tools=[], needs_llm=True, intent="casual")
            if q_lower.startswith(courtesy + " for"):
                return RoutingDecision(tools=[], needs_llm=True, intent="casual")
            match = re.match(rf"^{re.escape(courtesy)}(?:[\s,!.?;:-]+)(.+)$", q_lower)
            if match:
                q_lower = match.group(1).strip(" ,.!?;:-")
                break

        if any(q_lower == g or q_lower.startswith(g + " ") for g in greetings):
            return RoutingDecision(tools=[], needs_llm=True, intent="greeting")
            
        if any(q_lower == c or q_lower.startswith(c + " ") for c in casual):
            return RoutingDecision(tools=[], needs_llm=True, intent="casual")

        # 2. Knowledge Query Routing
        tools = []
        
        # Real-time / temporal / dynamic web triggers
        time_patterns = [
            r"\b20\d{2}\b", r"\btoday\b", r"\bnow\b", r"\bcurrent\b", 
            r"\brecent\b", r"\blatest\b", r"\bnews\b", r"\bweather\b", 
            r"\bstock\b", r"\bprice\b", r"\btonight\b", r"\byesterday\b"
        ]
        if any(re.search(p, q_lower) for p in time_patterns):
            tools.append("web_search")

        # Encyclopedic / historical triggers
        wiki_patterns = [
            "who is", "who was", "what is the history", "where is", 
            "biography", "born", "invented", "founded in", "capital of"
        ]
        if any(w in q_lower for w in wiki_patterns):
            tools.extend(["wikipedia", "web_search"])
            
        # Domain documentation / specific terminology triggers
        doc_patterns = [
            "explain", "how does", "what does", "define", "according to",
            "document", "uploaded", "pdf", "file", "policy", "manual", "report"
        ]
        if any(d in q_lower for d in doc_patterns):
            tools.append("document_search")

        # Fallback to parallel execution of web + docs if uncertain
        if not tools:
            tools = ["document_search", "web_search"]
            
        # Deduplicate while preserving order
        deduped_tools = list(dict.fromkeys(tools))

        return RoutingDecision(tools=deduped_tools, needs_llm=True, intent="knowledge")
