from flask import Flask, render_template, jsonify
from titanic_dataanalysis import perform_analysis
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/analysis')
def analysis():
    analysis_results = perform_analysis()
    return jsonify(analysis_results)
if __name__ == '__main__':
    app.run(debug=True)
