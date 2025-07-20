import os
from flask import Flask, jsonify

app = Flask(__name__)

# Railway port configuration
PORT = int(os.environ.get('PORT', 5000))

@app.route('/')
def home():
    return jsonify({
        "message": "Railway Flask API is working!", 
        "status": "running",
        "port": PORT,
        "test": "Basic Flask without YOLO"
    }), 200

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "document-processor-basic"
    }), 200

@app.route('/test')
def test():
    return "✅ Railway deployment successful - Flask is working!"

if __name__ == '__main__':
    print(f"🚀 Starting basic Flask app on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
    
