import React, { useState, useEffect, useRef } from 'react';

// --- CSS Styles ---
// All styling is now handled by this block of regular CSS.
const styles = `
  .chat-app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background-color: #f3f4f6;
    font-family: sans-serif;
  }
  .chat-header {
    background-color: #ffffff;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
  }
  .chat-header h1 {
    font-size: 1.5rem;
    font-weight: bold;
    color: #1f2937;
  }
  .chat-main {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem;
  }
  .message-container {
    display: flex;
    margin-bottom: 1rem;
  }
  .message-container.user {
    justify-content: flex-end;
  }
  .message-container.assistant {
    justify-content: flex-start;
  }
  .message-bubble {
    max-width: 80%;
    padding: 0.75rem 1rem;
    border-radius: 1.25rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    white-space: pre-wrap;
  }
  .message-bubble.user {
    background-color: #2563eb;
    color: #ffffff;
    border-bottom-right-radius: 0.25rem;
  }
  .message-bubble.assistant {
    background-color: #e5e7eb;
    color: #1f2937;
    border-bottom-left-radius: 0.25rem;
  }
  .loading-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .loading-dot {
    width: 0.5rem;
    height: 0.5rem;
    background-color: #6b7280;
    border-radius: 50%;
    animation: pulse 1.4s infinite ease-in-out both;
  }
  .loading-dot:nth-child(2) {
    animation-delay: 0.2s;
  }
  .loading-dot:nth-child(3) {
    animation-delay: 0.4s;
  }
  @keyframes pulse {
    0%, 80%, 100% {
      transform: scale(0);
    } 40% {
      transform: scale(1.0);
    }
  }
  .chat-footer {
    background-color: #ffffff;
    border-top: 1px solid #e5e7eb;
    padding: 1rem;
  }
  .chat-form {
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .chat-input {
    flex: 1;
    padding: 0.75rem;
    border: 1px solid #d1d5db;
    border-radius: 9999px;
    transition: box-shadow 0.2s;
  }
  .chat-input:focus {
    outline: none;
    box-shadow: 0 0 0 2px #3b82f6;
  }
  .send-button {
    padding: 0.75rem;
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    transition: background-color 0.2s;
  }
  .send-button:hover {
    background-color: #1d4ed8;
  }
  .send-button:disabled {
    background-color: #93c5fd;
    cursor: not-allowed;
  }
`;

// --- Helper Components ---

const SendIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M10 14L21 3M21 3L14.5 21C14.246 21.54 13.536 21.697 13.11 21.272L10 18M21 3L10 14L10 18M10 18L3.129 15.871C2.553 15.683 2.317 14.945 2.728 14.533L10 7L10 18Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const ChatMessage = ({ message }) => {
  const isUser = message.role === 'user';
  return (
    <div className={`message-container ${isUser ? 'user' : 'assistant'}`}>
      <div className={`message-bubble ${isUser ? 'user' : 'assistant'}`}>
        <p>{message.content}</p>
      </div>
    </div>
  );
};

// --- Main App Component ---

export default function App() {
  // --- State Management ---
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I am your AI Job Assistant. How can I help you find a job today?',
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);
  
  const DUMMY_USER_ID = "user_frontend_12345";

  // --- Effects ---
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // --- API Interaction ---
  const handleSendMessage = async (e) => {
    e.preventDefault();
    const userQuery = inputValue.trim();
    if (!userQuery) return;

    const newUserMessage = { role: 'user', content: userQuery };
    setMessages(prev => [...prev, newUserMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const requestBody = {
        user_id: DUMMY_USER_ID,
        query: userQuery,
        history: [...messages, newUserMessage] // Send the complete history
      };

      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      const assistantMessage = { role: 'assistant', content: data.response };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      console.error("Failed to send message:", error);
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // --- Render ---
  return (
    <>
      <style>{styles}</style>
      <div className="chat-app">
        <header className="chat-header">
          <h1>AI Job Assistant</h1>
        </header>

        <main className="chat-main">
          {messages.map((msg, index) => (
            <ChatMessage key={index} message={msg} />
          ))}
          {isLoading && (
            <div className="message-container assistant">
              <div className="message-bubble assistant">
                <div className="loading-indicator">
                  <div className="loading-dot"></div>
                  <div className="loading-dot"></div>
                  <div className="loading-dot"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </main>

        <footer className="chat-footer">
          <form onSubmit={handleSendMessage} className="chat-form">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Describe your ideal job..."
              className="chat-input"
              disabled={isLoading}
            />
            <button
              type="submit"
              className="send-button"
              disabled={isLoading || !inputValue.trim()}
            >
              <SendIcon />
            </button>
          </form>
        </footer>
      </div>
    </>
  );
}
