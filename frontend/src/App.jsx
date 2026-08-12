import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import ChatContainer from './components/ChatContainer';
import { streamChat, checkHealth } from './services/api';

export default function App() {
  const [sessions, setSessions] = useState(() => {
    try {
      const saved = localStorage.getItem('agentic_rag_sessions_local');
      if (!saved) return [];
      
      const parsed = JSON.parse(saved);
      const fifteenDaysMs = 15 * 24 * 60 * 60 * 1000;
      const now = Date.now();
      
      // Auto-delete sessions older than 15 days
      const filtered = parsed.filter((session) => {
        if (!session.createdAt) return true; // fallback for missing dates
        const sessionAge = now - new Date(session.createdAt).getTime();
        return sessionAge <= fifteenDaysMs;
      });
      
      return filtered;
    } catch {
      return [];
    }
  });
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isOnline, setIsOnline] = useState(true);

  // Responsive Sidebar States
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isMobile, setIsMobile] = useState(false);

  const abortControllerRef = useRef(null);
  // Guard flag: when true, skip the session→messages sync effect
  // so that handleSend's setMessages is not overwritten.
  const isSendingRef = useRef(false);

  // Detect Mobile Viewport
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (mobile) {
        setIsSidebarOpen(false);
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);



  // Sync sessions to localStorage when updated
  useEffect(() => {
    const storageKey = `agentic_rag_sessions_local`;
    try {
      localStorage.setItem(storageKey, JSON.stringify(sessions));
    } catch (e) {
      console.error('Failed to save sessions:', e);
    }
  }, [sessions]);

  // Check backend health on mount and poll periodically every 8s
  useEffect(() => {
    let isMounted = true;
    const verifyHealth = async () => {
      try {
        const health = await checkHealth();
        if (isMounted) {
          setIsOnline(health.status === 'ready' || health.status === 'healthy');
        }
      } catch {
        if (isMounted) {
          setIsOnline(false);
        }
      }
    };

    verifyHealth();
    const interval = setInterval(verifyHealth, 8000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // When user clicks a session in the sidebar, load its messages.
  // Skip when a send is in progress — handleSend manages messages directly.
  useEffect(() => {
    if (isSendingRef.current) return;

    if (activeSessionId) {
      const current = sessions.find((s) => s.id === activeSessionId);
      if (current) {
        setMessages(current.messages || []);
      }
    } else {
      setMessages([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSessionId]);

  // Toggle Sidebar
  const handleToggleSidebar = () => {
    setIsSidebarOpen((prev) => !prev);
  };

  // Close Mobile Sidebar
  const handleCloseMobile = () => {
    if (isMobile) {
      setIsSidebarOpen(false);
    }
  };

  // Create new session / Reset view
  const handleNewSession = () => {
    if (isLoading && abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    isSendingRef.current = false;
    setActiveSessionId(null);
    setMessages([]);
    setInput('');
  };

  // Delete session
  const handleDeleteSession = (sessionId) => {
    setSessions((prev) => prev.filter((s) => s.id !== sessionId));
    if (activeSessionId === sessionId) {
      setActiveSessionId(null);
      setMessages([]);
    }
  };

  // Clear current chat messages
  const handleClearChat = () => {
    if (!activeSessionId) {
      setMessages([]);
      return;
    }
    setMessages([]);
    setSessions((prev) =>
      prev.map((s) => (s.id === activeSessionId ? { ...s, messages: [] } : s))
    );
  };

  // Stop generation stream
  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsLoading(false);
      isSendingRef.current = false;
    }
  };

  // Send message
  const handleSend = async (overridePrompt) => {
    const promptToSend = (typeof overridePrompt === 'string' ? overridePrompt : input).trim();
    if (!promptToSend || isLoading) return;

    // Set guard so the session→messages sync effect doesn't overwrite us
    isSendingRef.current = true;

    let currentSessionId = activeSessionId;
    if (!currentSessionId) {
      currentSessionId = `conv_${Date.now()}`;
      const newSession = {
        id: currentSessionId,
        title: promptToSend.slice(0, 32) + (promptToSend.length > 32 ? '...' : ''),
        createdAt: new Date().toISOString(),
        messages: [],
      };
      setSessions((prev) => [newSession, ...prev]);
      setActiveSessionId(currentSessionId);
    } else {
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id === currentSessionId && (!s.messages || s.messages.length === 0)) {
            return {
              ...s,
              title: promptToSend.slice(0, 32) + (promptToSend.length > 32 ? '...' : ''),
            };
          }
          return s;
        })
      );
    }

    const userMessage = { role: 'user', content: promptToSend };
    const initialAssistantMessage = {
      id: null,
      role: 'assistant',
      content: '',
      tool_used: null,
      source_type: null,
      stepLabel: 'Analyzing context...',
      sources: [],
      latency_seconds: null,
    };

    const updatedMessages = [...messages, userMessage, initialAssistantMessage];
    setMessages(updatedMessages);
    setInput('');
    setIsLoading(true);

    abortControllerRef.current = new AbortController();

    const historyPayload = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    let accumulatedContent = '';
    let messageId = null;
    let latestSources = [];

    await streamChat({
      question: promptToSend,
      conversationId: currentSessionId,
      chatHistory: historyPayload,
      signal: abortControllerRef.current.signal,
      onStep: (stepData) => {
        setMessages((prev) => {
          const next = [...prev];
          const lastIdx = next.length - 1;
          if (lastIdx >= 0 && next[lastIdx].role === 'assistant') {
            next[lastIdx] = {
              ...next[lastIdx],
              stepLabel: stepData.label,
              tool_used: stepData.tool_used || next[lastIdx].tool_used,
            };
          }
          return next;
        });
      },
      onMetadata: (meta) => {
        messageId = meta.message_id || messageId;
        latestSources = meta.sources || [];
        setMessages((prev) => {
          const next = [...prev];
          const lastIdx = next.length - 1;
          if (lastIdx >= 0 && next[lastIdx].role === 'assistant') {
            next[lastIdx] = {
              ...next[lastIdx],
              id: meta.message_id || next[lastIdx].id,
              tool_used: meta.tool_used,
              source_type: meta.source_type,
              sources: meta.sources || [],
            };
          }
          return next;
        });
      },
      onToken: (token) => {
        accumulatedContent += token;
        setMessages((prev) => {
          const next = [...prev];
          const lastIdx = next.length - 1;
          if (lastIdx >= 0 && next[lastIdx].role === 'assistant') {
            next[lastIdx] = {
              ...next[lastIdx],
              content: accumulatedContent,
            };
          }
          return next;
        });
      },
      onDone: (doneData) => {
        const finalAnswer = (doneData.full_answer || accumulatedContent || '').trim();
        const finalMsgId = doneData.message_id || messageId;

        const finalAssistantMessage = {
          id: finalMsgId,
          role: 'assistant',
          content: finalAnswer,
          tool_used: doneData.tool_used,
          source_type: doneData.source_type,
          latency_seconds: doneData.latency_seconds,
          sources: latestSources,
          stepLabel: null,
        };

        setMessages((prev) => {
          const next = [...prev];
          const lastIdx = next.length - 1;
          if (lastIdx >= 0 && next[lastIdx].role === 'assistant') {
            next[lastIdx] = finalAssistantMessage;
          }
          return next;
        });

        // Persist final messages to session
        setSessions((prevSessions) =>
          prevSessions.map((s) => {
            if (s.id === currentSessionId) {
              return {
                ...s,
                messages: [
                  ...(s.messages || []),
                  userMessage,
                  finalAssistantMessage,
                ],
              };
            }
            return s;
          })
        );

        setIsLoading(false);
        // Release guard
        isSendingRef.current = false;
      },
      onError: (err) => {
        setMessages((prev) => {
          const next = [...prev];
          const lastIdx = next.length - 1;
          if (lastIdx >= 0 && next[lastIdx].role === 'assistant') {
            next[lastIdx] = {
              ...next[lastIdx],
              content: `Error: ${err}`,
              stepLabel: null,
            };
          }
          return next;
        });
        setIsLoading(false);
        isSendingRef.current = false;
      },
    });
  };

  return (
    <div className="flex h-screen w-screen bg-[#000000] text-white overflow-hidden font-sans">
      <Sidebar
        isOpen={isSidebarOpen}
        onToggle={handleToggleSidebar}
        isMobile={isMobile}
        onCloseMobile={handleCloseMobile}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={(id) => {
          isSendingRef.current = false;
          setActiveSessionId(id);
        }}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
      />

      <ChatContainer
        messages={messages}
        input={input}
        setInput={setInput}
        onSend={() => handleSend()}
        onStop={handleStop}
        isLoading={isLoading}
        onClearChat={handleClearChat}
        onNewSession={handleNewSession}
        isSidebarOpen={isSidebarOpen}
        onToggleSidebar={handleToggleSidebar}
        isOnline={isOnline}
      />
    </div>
  );
}
