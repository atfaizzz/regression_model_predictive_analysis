// chart.js

document.addEventListener('DOMContentLoaded', function () {
    // Retrieve data from Flask backend (replace with actual data)
    const data = {
        labels: ['January', 'February', 'March', 'April', 'May'],
        datasets: [{
            label: 'Sample Data',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1,
            data: [10, 20, 30, 40, 50],
        }]
    };

    // Define chart configuration
    const config = {
        type: 'bar',
        data: data,
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    };

    // Create and render the chart
    var myChart = new Chart(document.getElementById('myChart'), config);
});
