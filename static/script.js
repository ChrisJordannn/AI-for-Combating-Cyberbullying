document.getElementById('analyze').addEventListener('click', function () {
    const comment = document.getElementById('comment').value;
    if (comment.trim() === '') {
        alert('Please enter a comment.');
        return;
    }

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `comment=${encodeURIComponent(comment)}`
    })
    .then(response => response.json())
    .then(data => {
        const metricsDiv = document.getElementById('metrics');
        metricsDiv.innerHTML = ''; // Clear previous results

        // Sort metrics based on percentage value in descending order
        const sortedMetrics = Object.entries(data).sort((a, b) => b[1] - a[1]);

        sortedMetrics.forEach(([key, value]) => {
            const severity = Math.round(value * 100);

            const metricDiv = document.createElement('div');
            metricDiv.className = 'metric';

            const metricName = document.createElement('span');
            metricName.className = 'metric-name';
            metricName.textContent = `${key} - ${severity}%`; // Display percentage above the bar

            const progressContainer = document.createElement('div');
            progressContainer.className = 'progress-container';

            const progressBar = document.createElement('div');
            progressBar.className = `progress-bar ${key.toLowerCase().replace(' ', '-')}`;
            progressBar.style.width = `${severity}%`;

            progressContainer.appendChild(progressBar);
            metricDiv.appendChild(metricName);
            metricDiv.appendChild(progressContainer);
            metricsDiv.appendChild(metricDiv);
        });
    })
    .catch(error => console.error('Error:', error));
});
