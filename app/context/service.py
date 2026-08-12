import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from app.core.config import settings
from app.context.session import SessionMemory, Turn
from app.core.logging import logger

class ContextService:
    """Manages in-memory session history and heuristic query rewriting."""
    
    def __init__(self):
        self.sessions: Dict[str, SessionMemory] = {}
        
    def _get_or_create(self, session_id: str) -> SessionMemory:
        self.clear_expired_sessions()
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionMemory(session_id=session_id)
        return self.sessions[session_id]
        
    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Add a turn to the session buffer."""
        session = self._get_or_create(session_id)
        session.turns.append(Turn(role=role, content=content))
        session.last_activity = datetime.now()
        
    def restore_session(self, session_id: str, history: List[Dict[str, str]]) -> None:
        """Rehydrate a session from external storage (e.g., browser history)."""
        session = self._get_or_create(session_id)
        # Only restore if backend memory is empty
        if not session.turns and history:
            logger.info(f"Rehydrating session {session_id} from frontend history ({len(history)} turns)")
            for msg in history:
                role = msg.get("role")
                content = msg.get("content")
                if role and content:
                    session.turns.append(Turn(role=role, content=content))
        session.last_activity = datetime.now()
        
    def get_context_window(self, session_id: str, max_turns: int = 10) -> List[Dict[str, str]]:
        """Get the recent chat history formatted for LLMs."""
        session = self._get_or_create(session_id)
        recent_turns = session.turns[-max_turns:]
        return [{"role": t.role, "content": t.content} for t in recent_turns]
        
    def format_history_text(self, session_id: str, max_turns: int = 10) -> str:
        """Format history as a simple text block."""
        window = self.get_context_window(session_id, max_turns)
        if not window:
            return "No prior conversation."
            
        lines = []
        for msg in window:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
        
    async def rewrite_query(self, session_id: str, question: str) -> str:
        """
        Contextual query rewriter. Uses LLM to resolve pronouns and references against recent history.
        """
        session = self._get_or_create(session_id)
        session.last_activity = datetime.now()
        
        if len(session.turns) < 2:
            return question
            
        q_clean = question.strip()
        q_lower = q_clean.lower()
        
        # Check for pronoun references or follow-up indicators to trigger LLM
        pronoun_pattern = r"\b(he|she|it|they|his|her|its|their|this|that|these|those)\b"
        followup_phrases = ["tell me more", "explain", "give an example", "why", "how so", "elaborate", "continue", "what about", "who", "what", "where", "when", "how", "which"]
        
        has_pronoun = bool(re.search(pronoun_pattern, q_lower))
        is_followup = any(q_lower.startswith(phrase) for phrase in followup_phrases)
        is_short = len(q_clean.split()) <= 8
        
        if not (has_pronoun or (is_followup and is_short)):
            return question
            
        history_text = self.format_history_text(session_id, max_turns=6)
        
        from app.llm.gateway import gateway
        prompt = f"""Given the following conversation history, rewrite the user's latest question into a standalone, fully contextualized query.
If the question is already clear on its own, return it unchanged.
Do NOT answer the question, just rewrite it. Do not include quotes or prefixes.

Conversation History:
{history_text}

Latest User Question: {q_clean}

Standalone Query:"""

        try:
            rewritten = await gateway.complete([{"role": "user", "content": prompt}])
            rewritten = rewritten.strip(' "\'')
            if rewritten and rewritten.lower() != q_lower:
                logger.info(f"ContextService LLM resolved: '{question}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"Failed to contextualize query: {e}")
            
        return question

    def clear_session(self, session_id: str) -> None:
        """Clear a session from memory."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def clear_expired_sessions(self) -> None:
        """Remove sessions that have exceeded the inactivity timeout."""
        now = datetime.now()
        timeout = timedelta(minutes=settings.SESSION_TIMEOUT_MINUTES)
        expired_ids = [
            sid for sid, session in self.sessions.items()
            if (now - session.last_activity) > timeout
        ]
        for sid in expired_ids:
            logger.info(f"Clearing expired session: {sid}")
            del self.sessions[sid]

# Singleton
context_service = ContextService()
