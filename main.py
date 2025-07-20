import os
from flask import Flask, jsonify

app = Flask(__name__)

# Download weights if they don't exist
def download_weights():
    if not os.path.exists("weights.pt"):
        weights_url = os.getenv('WEIGHTS_URL')
        if weights_url:
            try:
                import requests
                print(f"[INFO] Downloading weights from {weights_url}")
                response = requests.get(weights_url, stream=True)
                with open('weights.pt', 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("[SUCCESS] Weights downloaded successfully")
                return True
            except Exception as e:
                print(f"[ERROR] Download failed: {e}")
                return False
        else:
            print("[INFO] No WEIGHTS_URL provided")
            return False
    return True

@app.route('/')
def home():
    return jsonify({
        'message': 'Document Processing API is running',
        'status': 'healthy'
    })

@app.route('/health')
def health():
    # Try to download weights if they don't exist
    download_weights()
    
    return jsonify({
        'status': 'healthy',
        'weights_exist': os.path.exists('weights.pt'),
        'weights_url_configured': os.getenv('WEIGHTS_URL') is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)
