"""
CSS Styles for SkinProAI components
"""

MAIN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* Hide Gradio footer */
.gradio-container footer { display: none !important; }

/* ============================================
   PATIENT SELECTION PAGE
   ============================================ */

.patient-select-container {
    min-height: 80vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
}

.patient-select-title {
    font-size: 32px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 8px;
    text-align: center;
}

.patient-select-subtitle {
    font-size: 16px;
    color: #6b7280;
    margin-bottom: 40px;
    text-align: center;
}

.patient-grid {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    justify-content: center;
    max-width: 800px;
}

.patient-card {
    background: white !important;
    border: 2px solid #e5e7eb !important;
    border-radius: 16px !important;
    padding: 24px 32px !important;
    min-width: 200px !important;
    cursor: pointer;
    transition: all 0.2s ease !important;
}

.patient-card:hover {
    border-color: #6366f1 !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15) !important;
    transform: translateY(-2px);
}

.new-patient-btn {
    background: #6366f1 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 32px !important;
    font-weight: 500 !important;
    margin-top: 24px;
}

.new-patient-btn:hover {
    background: #4f46e5 !important;
}

/* ============================================
   ANALYSIS PAGE - MAIN LAYOUT
   ============================================ */

.analysis-container {
    display: flex;
    height: calc(100vh - 80px);
    min-height: 600px;
}

/* Sidebar */
.query-sidebar {
    width: 280px;
    background: #f9fafb;
    border-right: 1px solid #e5e7eb;
    padding: 20px;
    overflow-y: auto;
    flex-shrink: 0;
}

.sidebar-header {
    font-size: 14px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #e5e7eb;
}

.query-item {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.15s;
}

.query-item:hover {
    border-color: #6366f1;
    background: #f5f3ff;
}

.query-item-title {
    font-size: 13px;
    font-weight: 500;
    color: #111827;
    margin-bottom: 4px;
}

.query-item-meta {
    font-size: 11px;
    color: #6b7280;
}

/* Main content area */
.main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 24px;
    overflow: hidden;
}

/* ============================================
   INPUT AREA (Greeting style)
   ============================================ */

.input-greeting {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
}

.greeting-title {
    font-size: 24px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 8px;
}

.greeting-subtitle {
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 32px;
}

.input-box-container {
    width: 100%;
    max-width: 600px;
    background: white;
    border: 2px solid #e5e7eb;
    border-radius: 16px;
    padding: 20px;
    transition: border-color 0.2s;
}

.input-box-container:focus-within {
    border-color: #6366f1;
}

.message-input textarea {
    border: none !important;
    resize: none !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
    padding: 0 !important;
}

.message-input textarea:focus {
    box-shadow: none !important;
}

.input-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid #f3f4f6;
}

.upload-btn {
    background: #f3f4f6 !important;
    color: #374151 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-size: 13px !important;
}

.upload-btn:hover {
    background: #e5e7eb !important;
}

.send-btn {
    background: #6366f1 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 500 !important;
}

.send-btn:hover {
    background: #4f46e5 !important;
}

.send-btn:disabled {
    background: #d1d5db !important;
    cursor: not-allowed;
}

/* Image preview */
.image-preview {
    margin-top: 16px;
    border-radius: 12px;
    overflow: hidden;
    max-height: 200px;
}

.image-preview img {
    max-height: 200px;
    object-fit: contain;
}

/* ============================================
   CHAT/RESULTS VIEW
   ============================================ */

.chat-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.results-area {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    margin-bottom: 16px;
}

/* Analysis output styling */
.analysis-output {
    line-height: 1.6;
    color: #333;
}

.stage {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    font-weight: 500;
    color: #1a1a1a;
    margin-top: 12px;
}

.stage-indicator {
    width: 8px;
    height: 8px;
    background: #6366f1;
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

.thinking {
    color: #6b7280;
    font-style: italic;
    font-size: 13px;
    padding: 4px 0 4px 16px;
    border-left: 2px solid #e5e7eb;
    margin: 4px 0;
}

.observation {
    color: #374151;
    font-size: 13px;
    padding: 4px 0 4px 16px;
}

.tool-output {
    background: #f8fafc;
    border-radius: 8px;
    margin: 12px 0;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}

.tool-header {
    background: #f1f5f9;
    padding: 8px 12px;
    font-weight: 500;
    font-size: 13px;
    color: #475569;
    border-bottom: 1px solid #e2e8f0;
}

.tool-content {
    padding: 12px;
    margin: 0;
    font-family: 'SF Mono', Monaco, monospace !important;
    font-size: 12px;
    line-height: 1.5;
    white-space: pre-wrap;
    color: #334155;
}

.result {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 12px 0;
    font-weight: 500;
    color: #065f46;
}

.error {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #b91c1c;
}

.response {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px;
    margin: 16px 0;
    line-height: 1.7;
}

.response ul, .response ol {
    margin: 8px 0;
    padding-left: 24px;
}

.response li {
    margin: 4px 0;
}

.complete {
    color: #6b7280;
    font-size: 12px;
    padding: 8px 0;
    text-align: center;
}

/* Confirmation */
.confirm-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 16px;
    margin: 16px 0;
    text-align: center;
}

.confirm-buttons {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 8px;
    padding: 12px;
    margin-top: 12px;
}

/* References */
.references {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    margin: 16px 0;
    overflow: hidden;
}

.references-header {
    background: #f3f4f6;
    padding: 8px 12px;
    font-weight: 500;
    font-size: 13px;
    border-bottom: 1px solid #e5e7eb;
}

.references ul {
    list-style: none;
    padding: 12px;
    margin: 0;
}

.ref-link {
    color: #6366f1;
    text-decoration: none;
    font-size: 13px;
}

.ref-link:hover {
    text-decoration: underline;
}

/* GradCAM */
.gradcam-inline {
    margin: 16px 0;
    background: #f8fafc;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}

.gradcam-header {
    background: #f1f5f9;
    padding: 8px 12px;
    font-weight: 500;
    font-size: 13px;
    border-bottom: 1px solid #e2e8f0;
}

.gradcam-inline img {
    max-width: 100%;
    max-height: 300px;
    display: block;
    margin: 12px auto;
}

/* Chat input at bottom */
.chat-input-area {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 12px 16px;
    display: flex;
    gap: 12px;
    align-items: flex-end;
}

.chat-input-area textarea {
    flex: 1;
    border: none !important;
    resize: none !important;
    font-size: 14px !important;
}

/* ============================================
   HEADER
   ============================================ */

.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    border-bottom: 1px solid #e5e7eb;
    background: white;
}

.app-title {
    font-size: 20px;
    font-weight: 600;
    color: #111827;
}

.back-btn {
    background: transparent !important;
    color: #6b7280 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-size: 13px !important;
}

.back-btn:hover {
    background: #f9fafb !important;
    color: #111827 !important;
}
"""
