import os
import sys
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from rag_engine import PureRAGEngine

load_dotenv()

app = Flask(__name__)
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = PureRAGEngine()
    return engine

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    use_rag = data.get('use_rag', True) # Default to True
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        result = get_engine().generate(question, use_rag=use_rag)
        return jsonify({
            "answer": result['answer'],
            "sources": result['sources']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=80, debug=True)
