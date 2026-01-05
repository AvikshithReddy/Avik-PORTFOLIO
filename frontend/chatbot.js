/**
 * Portfolio Chatbot Widget
 * Connects to the FastAPI backend for RAG-powered responses
 */

class PortfolioChatbot {
    constructor(config = {}) {
        this.config = {
            backendUrl: config.backendUrl || 'http://localhost:8000',
            theme: config.theme || 'light',
            position: config.position || 'bottom-right',
            includeSources: config.includeSources !== false,
            ...config
        };
        
        this.sessionId = this.loadSessionId();
        this.isOpen = false;
        this.messageHistory = [];
        
        this.init();
    }
    
    init() {
        this.createChatWidget();
        this.attachEventListeners();
        this.checkBackendHealth();
    }
    
    loadSessionId() {
        let sessionId = localStorage.getItem('chatbot_session_id');
        if (!sessionId) {
            sessionId = this.generateUUID();
            localStorage.setItem('chatbot_session_id', sessionId);
        }
        return sessionId;
    }
    
    generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
    
    createChatWidget() {
        const widgetHTML = `
            <div id="chatbot-container" class="chatbot-container ${this.config.position}">
                <!-- Chat Button -->
                <button id="chatbot-toggle" class="chatbot-toggle" aria-label="Toggle chat">
                    <svg class="icon-chat" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
                    </svg>
                    <svg class="icon-close" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
                
                <!-- Chat Window -->
                <div id="chatbot-window" class="chatbot-window">
                    <div class="chatbot-header">
                        <h3>Avikshith's Portfolio Assistant</h3>
                        <p class="chatbot-subtitle">Ask me about projects, skills, and experience</p>
                    </div>
                    
                    <div id="chatbot-messages" class="chatbot-messages">
                        <div class="message bot-message">
                            <div class="message-content">
                                Hello! I'm Avikshith's portfolio assistant. I can answer questions about his projects, skills, experience, and background. What would you like to know?
                            </div>
                        </div>
                    </div>
                    
                    <div class="chatbot-input-area">
                        <div id="chatbot-status" class="chatbot-status"></div>
                        <div class="chatbot-input-wrapper">
                            <input 
                                type="text" 
                                id="chatbot-input" 
                                placeholder="Ask about projects, skills, experience..."
                                autocomplete="off"
                            />
                            <button id="chatbot-send" class="chatbot-send-btn" aria-label="Send message">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                    <line x1="22" y1="2" x2="11" y2="13"></line>
                                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', widgetHTML);
        this.injectStyles();
    }
    
    attachEventListeners() {
        const toggleBtn = document.getElementById('chatbot-toggle');
        const sendBtn = document.getElementById('chatbot-send');
        const input = document.getElementById('chatbot-input');
        
        toggleBtn.addEventListener('click', () => this.toggleChat());
        sendBtn.addEventListener('click', () => this.sendMessage());
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
    }
    
    toggleChat() {
        this.isOpen = !this.isOpen;
        const container = document.getElementById('chatbot-container');
        const window = document.getElementById('chatbot-window');
        
        if (this.isOpen) {
            container.classList.add('open');
            window.style.display = 'flex';
            document.getElementById('chatbot-input').focus();
        } else {
            container.classList.remove('open');
            setTimeout(() => {
                window.style.display = 'none';
            }, 300);
        }
    }
    
    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.config.backendUrl}/health`);
            const data = await response.json();
            
            if (!data.rag_index_loaded) {
                this.showStatus('⚠️ Backend index not loaded. Limited functionality.', 'warning');
            }
        } catch (error) {
            console.error('Backend health check failed:', error);
            this.showStatus('⚠️ Cannot connect to backend', 'error');
        }
    }
    
    async sendMessage() {
        const input = document.getElementById('chatbot-input');
        const query = input.value.trim();
        
        if (!query) return;
        
        // Add user message to UI
        this.addMessage(query, 'user');
        input.value = '';
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            const response = await fetch(`${this.config.backendUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    session_id: this.sessionId,
                    include_sources: this.config.includeSources
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Remove typing indicator
            this.removeTypingIndicator();
            
            // Add bot response
            this.addMessage(data.response, 'bot', {
                confidence: data.confidence,
                sources: data.sources || []
            });
            
            // Update session ID
            this.sessionId = data.session_id;
            
        } catch (error) {
            console.error('Chat error:', error);
            this.removeTypingIndicator();
            this.addMessage(
                'Sorry, I encountered an error processing your request. Please try again.',
                'bot',
                { isError: true }
            );
        }
    }
    
    addMessage(text, sender, metadata = {}) {
        const messagesContainer = document.getElementById('chatbot-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        let messageHTML = `<div class="message-content">${this.escapeHtml(text)}</div>`;
        
        // Add sources if available
        if (metadata.sources && metadata.sources.length > 0) {
            messageHTML += '<div class="message-sources">';
            messageHTML += '<div class="sources-header">Sources:</div>';
            metadata.sources.slice(0, 3).forEach(source => {
                messageHTML += `
                    <div class="source-item">
                        <span class="source-type">${source.source_type}</span>
                        <span class="source-name">${this.escapeHtml(source.source_name)}</span>
                    </div>
                `;
            });
            messageHTML += '</div>';
        }
        
        // Add confidence indicator for bot messages
        if (sender === 'bot' && metadata.confidence !== undefined) {
            const confidenceClass = metadata.confidence > 0.7 ? 'high' : metadata.confidence > 0.5 ? 'medium' : 'low';
            messageHTML += `<div class="confidence-indicator ${confidenceClass}" title="Confidence: ${(metadata.confidence * 100).toFixed(0)}%"></div>`;
        }
        
        messageDiv.innerHTML = messageHTML;
        messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        this.messageHistory.push({ text, sender, metadata });
    }
    
    showTypingIndicator() {
        const messagesContainer = document.getElementById('chatbot-messages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message typing-indicator';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-content">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        `;
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) indicator.remove();
    }
    
    showStatus(message, type = 'info') {
        const statusDiv = document.getElementById('chatbot-status');
        statusDiv.textContent = message;
        statusDiv.className = `chatbot-status ${type}`;
        statusDiv.style.display = 'block';
        
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 5000);
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    injectStyles() {
        const styles = `
            .chatbot-container {
                position: fixed;
                z-index: 9999;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            .chatbot-container.bottom-right {
                bottom: 20px;
                right: 20px;
            }
            
            .chatbot-toggle {
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
            }
            
            .chatbot-toggle:hover {
                transform: scale(1.05);
                box-shadow: 0 6px 16px rgba(0,0,0,0.2);
            }
            
            .chatbot-toggle svg {
                color: white;
                stroke-width: 2;
            }
            
            .chatbot-toggle .icon-close {
                display: none;
            }
            
            .chatbot-container.open .chatbot-toggle .icon-chat {
                display: none;
            }
            
            .chatbot-container.open .chatbot-toggle .icon-close {
                display: block;
            }
            
            .chatbot-window {
                position: absolute;
                bottom: 80px;
                right: 0;
                width: 380px;
                max-width: calc(100vw - 40px);
                height: 550px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                display: none;
                flex-direction: column;
                overflow: hidden;
                animation: slideUp 0.3s ease;
            }
            
            @keyframes slideUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .chatbot-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 12px 12px 0 0;
            }
            
            .chatbot-header h3 {
                margin: 0;
                font-size: 18px;
                font-weight: 600;
            }
            
            .chatbot-subtitle {
                margin: 5px 0 0;
                font-size: 13px;
                opacity: 0.9;
            }
            
            .chatbot-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #f8f9fa;
            }
            
            .message {
                margin-bottom: 16px;
                animation: fadeIn 0.3s ease;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .message-content {
                padding: 12px 16px;
                border-radius: 12px;
                max-width: 85%;
                word-wrap: break-word;
                position: relative;
            }
            
            .user-message .message-content {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                margin-left: auto;
                border-bottom-right-radius: 4px;
            }
            
            .bot-message .message-content {
                background: white;
                color: #333;
                border: 1px solid #e0e0e0;
                border-bottom-left-radius: 4px;
            }
            
            .typing-indicator .message-content {
                padding: 16px 20px;
                display: flex;
                gap: 4px;
            }
            
            .typing-dot {
                width: 8px;
                height: 8px;
                background: #999;
                border-radius: 50%;
                animation: typing 1.4s infinite;
            }
            
            .typing-dot:nth-child(2) { animation-delay: 0.2s; }
            .typing-dot:nth-child(3) { animation-delay: 0.4s; }
            
            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
                30% { transform: translateY(-10px); opacity: 1; }
            }
            
            .message-sources {
                margin-top: 8px;
                padding: 8px;
                background: #f0f0f0;
                border-radius: 6px;
                font-size: 12px;
            }
            
            .sources-header {
                font-weight: 600;
                margin-bottom: 4px;
                color: #666;
            }
            
            .source-item {
                padding: 4px 0;
                display: flex;
                gap: 8px;
            }
            
            .source-type {
                background: #667eea;
                color: white;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 10px;
                text-transform: uppercase;
            }
            
            .source-name {
                color: #555;
            }
            
            .confidence-indicator {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                position: absolute;
                bottom: 4px;
                right: 4px;
            }
            
            .confidence-indicator.high { background: #4caf50; }
            .confidence-indicator.medium { background: #ff9800; }
            .confidence-indicator.low { background: #f44336; }
            
            .chatbot-input-area {
                border-top: 1px solid #e0e0e0;
                background: white;
                padding: 16px;
            }
            
            .chatbot-status {
                display: none;
                padding: 8px;
                margin-bottom: 8px;
                border-radius: 6px;
                font-size: 12px;
            }
            
            .chatbot-status.warning {
                background: #fff3cd;
                color: #856404;
                border: 1px solid #ffeeba;
            }
            
            .chatbot-status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            
            .chatbot-input-wrapper {
                display: flex;
                gap: 8px;
            }
            
            #chatbot-input {
                flex: 1;
                padding: 12px 16px;
                border: 1px solid #e0e0e0;
                border-radius: 24px;
                font-size: 14px;
                outline: none;
                transition: border-color 0.3s;
            }
            
            #chatbot-input:focus {
                border-color: #667eea;
            }
            
            .chatbot-send-btn {
                width: 44px;
                height: 44px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s;
            }
            
            .chatbot-send-btn:hover {
                transform: scale(1.05);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }
            
            .chatbot-send-btn svg {
                color: white;
                stroke-width: 2;
            }
            
            /* Mobile Responsive */
            @media (max-width: 480px) {
                .chatbot-window {
                    width: calc(100vw - 40px);
                    height: 500px;
                }
            }
        `;
        
        const styleSheet = document.createElement('style');
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }
}

// Auto-initialize chatbot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize with default configuration
    // Can be customized by setting window.chatbotConfig before this script loads
    const config = window.chatbotConfig || {};
    window.portfolioChatbot = new PortfolioChatbot(config);
});
