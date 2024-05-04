from flask import Flask, render_template, jsonify
from titanic_dataanalysis import perform_analysis  # Import perform_analysis function from titanic_dataanalysis.py

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    # Call the perform_analysis function to get analysis results
    analysis_results = perform_analysis()

    # Return the analysis results as JSON
    return jsonify(analysis_results)

if __name__ == '__main__':
    app.run(debug=True)
