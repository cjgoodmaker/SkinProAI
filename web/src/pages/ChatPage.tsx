import { useEffect, useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import { streamChatMessage } from '../services/streaming';
import { ToolCallCard } from '../components/ToolCallCard';
import { MessageContent } from '../components/MessageContent';
import { Patient, ChatMessage, ToolCall } from '../types';
import './ChatPage.css';

function formatTime(ts: string) {
  return new Date(ts).toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
  });
}

export function ChatPage() {
  const { patientId } = useParams<{ patientId: string }>();
  const navigate = useNavigate();

  const [patient, setPatient] = useState<Patient | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!patientId) return;
    api.getPatient(patientId).then(res => setPatient(res.patient));
    api.getChatHistory(patientId).then(res => setMessages(res.messages ?? []));
  }, [patientId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
    setImagePreview(URL.createObjectURL(file));
  };

  const handleSend = async () => {
    if ((!input.trim() && !selectedImage) || !patientId || isStreaming) return;

    const userMsgId = `msg-${Date.now()}`;
    const assistantMsgId = `msg-${Date.now() + 1}`;

    const userMsg: ChatMessage = {
      id: userMsgId,
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
      image_url: imagePreview ?? undefined,
    };

    const assistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      tool_calls: [],
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);

    const imgToSend = selectedImage;
    const contentToSend = input;
    setInput('');
    setSelectedImage(null);
    setImagePreview(null);
    setIsStreaming(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    await streamChatMessage(patientId, contentToSend, imgToSend, {
      onText: (chunk) => {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantMsgId ? { ...m, content: m.content + chunk } : m
          )
        );
      },
      onToolStart: (tool, callId) => {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantMsgId
              ? {
                  ...m,
                  tool_calls: [
                    ...(m.tool_calls ?? []),
                    { id: callId, tool, status: 'calling' as const },
                  ],
                }
              : m
          )
        );
      },
      onToolResult: (_tool, callId, result) => {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantMsgId
              ? {
                  ...m,
                  tool_calls: (m.tool_calls ?? []).map(tc =>
                    tc.id === callId
                      ? { ...tc, status: 'complete' as const, result: result as ToolCall['result'] }
                      : tc
                  ),
                }
              : m
          )
        );
      },
      onDone: () => setIsStreaming(false),
      onError: (err) => {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantMsgId ? { ...m, content: `[ERROR]${err}[/ERROR]` } : m
          )
        );
        setIsStreaming(false);
      },
    });
  };

  const handleClear = async () => {
    if (!patientId || !confirm('Clear chat history?')) return;
    await api.clearChat(patientId);
    setMessages([]);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
  };

  return (
    <div className="chat-page">
      {/* Header */}
      <header className="chat-header">
        <button className="header-back-btn" onClick={() => navigate('/')} title="Back to patients">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z" />
          </svg>
        </button>
        <div className="header-center">
          <span className="header-app-name">SkinProAI</span>
          {patient && <span className="header-patient-name">{patient.name}</span>}
        </div>
        <button className="header-clear-btn" onClick={handleClear} title="Clear history">
          Clear
        </button>
      </header>

      {/* Messages */}
      <main className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-icon">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z" />
              </svg>
            </div>
            <p>Send a message or attach a skin image to begin analysis.</p>
          </div>
        )}

        {messages.map(msg => (
          <div key={msg.id} className={`message-row ${msg.role}`}>
            {msg.role === 'user' ? (
              <div className="user-message">
                <div className="user-bubble">
                  {msg.image_url && (
                    <img src={msg.image_url} className="message-image" alt="Attached" />
                  )}
                  {msg.content && <p className="bubble-text">{msg.content}</p>}
                </div>
                <span className="msg-time">{formatTime(msg.timestamp)}</span>
              </div>
            ) : (
              <div className="assistant-message">
                {/* Tool call status lines */}
                {(msg.tool_calls ?? []).map(tc => (
                  <ToolCallCard key={tc.id} toolCall={tc} />
                ))}

                {/* Text content */}
                {msg.content ? (
                  <div className="assistant-text">
                    <MessageContent text={msg.content} />
                  </div>
                ) : (!msg.tool_calls || msg.tool_calls.length === 0) && isStreaming ? (
                  <div className="thinking">
                    <span className="dot" />
                    <span className="dot" />
                    <span className="dot" />
                  </div>
                ) : null}

                {(msg.content || (msg.tool_calls && msg.tool_calls.length > 0)) && (
                  <span className="msg-time">{formatTime(msg.timestamp)}</span>
                )}
              </div>
            )}
          </div>
        ))}

        <div ref={messagesEndRef} />
      </main>

      {/* Input bar */}
      <footer className="chat-input-bar">
        {imagePreview && (
          <div className="image-preview-container">
            <img src={imagePreview} alt="Preview" className="image-preview-thumb" />
            <button
              className="remove-image-btn"
              onClick={() => { setSelectedImage(null); setImagePreview(null); }}
              title="Remove image"
            >
              ×
            </button>
          </div>
        )}
        <div className="input-row">
          <button
            className="attach-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Attach image"
            disabled={isStreaming}
          >
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5a2.5 2.5 0 015 0v10.5c0 .83-.67 1.5-1.5 1.5s-1.5-.67-1.5-1.5V6H10v9.5a2.5 2.5 0 005 0V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z" />
            </svg>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={e => e.target.files?.[0] && handleImageSelect(e.target.files[0])}
          />
          <textarea
            ref={textareaRef}
            className="chat-input"
            placeholder="Type a message..."
            value={input}
            onChange={handleTextareaChange}
            onKeyDown={handleKeyDown}
            disabled={isStreaming}
            rows={1}
          />
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={isStreaming || (!input.trim() && !selectedImage)}
            title="Send"
          >
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          </button>
        </div>
      </footer>
    </div>
  );
}
