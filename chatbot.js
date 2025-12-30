class PortfolioChatbot {
  constructor(options = {}) {
    this.apiUrl = options.apiUrl || 'https://avikshithreddy.github.io/Avik-PORTFOLIO/';
    this.sessionId = this.getOrCreateSessionId();
    this.isOpen = false;
    this.isLoading = false;
    this.messages = [];
    
    // Wait a moment then initialize
    setTimeout(() => this.init(), 100);
  }

  getOrCreateSessionId() {
    let sessionId = localStorage.getItem('portfolio-chat-session');
    if (!sessionId) {
      sessionId = 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('portfolio-chat-session', sessionId);
    }
    return sessionId;
  }

  init() {
    console.log('🚀 Initializing Portfolio Chatbot');
    this.createChatbotUI();
    this.attachEventListeners();
    console.log('✅ Chatbot initialized');
  }

  createChatbotUI() {
    console.log('🎨 createChatbotUI() called');
    
    // Check if already exists
    if (document.getElementById('portfolio-chatbot-container')) {
      console.log('⚠️  Chatbot UI already exists, skipping');
      return;
    }
    
    try {
      const container = document.createElement('div');
      container.id = 'portfolio-chatbot-container';
      console.log('✅ Container created');
      
      // Create widget with button
      const widget = document.createElement('div');
      widget.id = 'chatbot-widget';
      widget.className = 'chatbot-widget';
      
      const button = document.createElement('button');
      button.id = 'chatbot-toggle';
      button.className = 'chatbot-toggle';
      button.title = 'Ask about my portfolio';
      button.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>`;
      console.log('✅ Button created');
      
      widget.appendChild(button);
      container.appendChild(widget);
      
      // Create chat window
      const chatWindow = document.createElement('div');
      chatWindow.id = 'chatbot-window';
      chatWindow.className = 'chatbot-window hidden';
      console.log('✅ Chat window created with class:', chatWindow.className);
      
      // Header
      const header = document.createElement('div');
      header.className = 'chatbot-header';
      header.innerHTML = `<h3>Avikshith's AI Assistant</h3><button id="chatbot-close" class="chatbot-close"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg></button>`;
      
      // Messages
      const messages = document.createElement('div');
      messages.id = 'chatbot-messages';
      messages.className = 'chatbot-messages';
      
      // Input
      const input = document.createElement('div');
      input.className = 'chatbot-input-area';
      input.innerHTML = `<input type="text" id="chatbot-input" class="chatbot-input" placeholder="Ask me anything..." autocomplete="off"/><button id="chatbot-send" class="chatbot-send" title="Send message"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg></button>`;
      
      chatWindow.appendChild(header);
      chatWindow.appendChild(messages);
      chatWindow.appendChild(input);
      container.appendChild(chatWindow);
      
      document.body.appendChild(container);
      console.log('✅ UI appended to body');
      console.log('🔍 Chatbot container in DOM:', !!document.getElementById('portfolio-chatbot-container'));
      console.log('🔍 Toggle button in DOM:', !!document.getElementById('chatbot-toggle'));
      console.log('🔍 Chat window in DOM:', !!document.getElementById('chatbot-window'));
    } catch (error) {
      console.error('❌ Error creating chatbot UI:', error);
    }
  }

  attachEventListeners() {
    console.log('🔗 attachEventListeners() called');
    
    const toggle = document.getElementById('chatbot-toggle');
    const close = document.getElementById('chatbot-close');
    const send = document.getElementById('chatbot-send');
    const input = document.getElementById('chatbot-input');
    
    console.log('🔍 Elements found:');
    console.log('  - toggle:', !!toggle);
    console.log('  - close:', !!close);
    console.log('  - send:', !!send);
    console.log('  - input:', !!input);
    
    if (toggle) {
      toggle.addEventListener('click', (e) => {
        console.log('📌 Toggle CLICKED!');
        e.preventDefault();
        e.stopPropagation();
        this.toggleChat();
      });
      console.log('✅ Toggle click listener attached');
    } else {
      console.error('❌ Toggle button not found!');
    }
    
    if (close) {
      close.addEventListener('click', (e) => {
        console.log('📌 Close CLICKED!');
        e.preventDefault();
        e.stopPropagation();
        this.closeChat();
      });
      console.log('✅ Close click listener attached');
    } else {
      console.error('❌ Close button not found!');
    }
    
    if (send) {
      send.addEventListener('click', (e) => {
        console.log('📌 Send CLICKED!');
        e.preventDefault();
        e.stopPropagation();
        this.sendMessage();
      });
      console.log('✅ Send click listener attached');
    } else {
      console.error('❌ Send button not found!');
    }
    
    if (input) {
      input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          this.sendMessage();
        }
      });
      console.log('✅ Input keypress listener attached');
    } else {
      console.error('❌ Input not found!');
    }
  }

  toggleChat() {
    if (this.isOpen) {
      this.closeChat();
    } else {
      this.openChat();
    }
  }

  openChat() {
    console.log('🔓 Opening chat');
    this.isOpen = true;
    const w = document.getElementById('chatbot-window');
    if (w) w.classList.remove('hidden');
    const i = document.getElementById('chatbot-input');
    if (i) i.focus();
  }

  closeChat() {
    console.log('🔒 Closing chat');
    this.isOpen = false;
    const w = document.getElementById('chatbot-window');
    if (w) w.classList.add('hidden');
  }

  async sendMessage() {
    const input = document.getElementById('chatbot-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    input.value = '';
    this.addMessage(message, true);
    
    this.isLoading = true;
    const loadingEl = document.createElement('div');
    loadingEl.className = 'chatbot-message assistant loading';
    loadingEl.textContent = 'Thinking...';
    document.getElementById('chatbot-messages').appendChild(loadingEl);
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);
      
      const response = await fetch(`${this.apiUrl}/api/chat`, {
        method: 'POST',
        mode: 'cors',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message,
          session_id: this.sessionId,
          include_sources: true
        }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      loadingEl.remove();
      this.addMessage(data.response, false);
      this.sessionId = data.session_id;
      
    } catch (error) {
      loadingEl.remove();
      const errorMsg = error.name === 'AbortError' 
        ? 'Request timed out. Please try again.'
        : `Error: ${error.message}`;
      this.addMessage(errorMsg, false);
      console.error('Chat error:', error);
    } finally {
      this.isLoading = false;
    }
  }

  addMessage(text, isUser) {
    const messagesDiv = document.getElementById('chatbot-messages');
    const messageEl = document.createElement('div');
    messageEl.className = `chatbot-message ${isUser ? 'user' : 'assistant'}`;
    messageEl.textContent = text;
    messagesDiv.appendChild(messageEl);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    this.messages.push({ role: isUser ? 'user' : 'assistant', content: text });
  }
}

// Auto-initialize on page load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    const backendUrl = window.location.hostname === 'localhost' 
      ? 'http://localhost:8000'
      : window.location.origin;
    if (!window.portfolioChatbot) {
      window.portfolioChatbot = new PortfolioChatbot({ apiUrl: backendUrl });
    }
  });
} else {
  const backendUrl = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000'
    : window.location.origin;
  if (!window.portfolioChatbot) {
    window.portfolioChatbot = new PortfolioChatbot({ apiUrl: backendUrl });
  }
}
