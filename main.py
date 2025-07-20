import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({'message': 'API is running', 'status': 'healthy'})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'weights_exist': os.path.exists('weights.pt')})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
