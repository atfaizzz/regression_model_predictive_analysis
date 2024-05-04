document.addEventListener('DOMContentLoaded', function () {
    // Make a GET request to the new route to retrieve analysis results
    fetch('/analysis')
        .then(response => response.json())
        .then(data => {
            // Process and use the analysis results as needed
            console.log('Analysis results:', data);
            // Example: Update HTML elements with analysis results
            document.getElementById('accuracy').innerText = `Accuracy: ${data['Logistic Regression']['Accuracy']}`;
            document.getElementById('precision').innerText = `Precision: ${data['Logistic Regression']['Precision']}`;
            document.getElementById('recall').innerText = `Recall: ${data['Logistic Regression']['Recall']}`;
            document.getElementById('f1-score').innerText = `F1-score: ${data['Logistic Regression']['F1-score']}`;
        })
        .catch(error => console.error('Error fetching analysis results:', error));
});
