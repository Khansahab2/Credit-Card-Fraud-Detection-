// static/script.js
document.getElementById('fraudForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData);
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');

    loading.style.display = 'block';
    result.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const res = await response.json();
        if (response.ok) {
            displayResult(res);
        } else {
            alert(res.error || 'Error occurred');
        }
    } catch (err) {
        alert('Network error: ' + err.message);
    } finally {
        loading.style.display = 'none';
    }
});

function displayResult(res) {
    const resultDiv = document.getElementById('result');
    const className = res.fraud ? 'fraud' : 'safe';
    const icon = res.fraud ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle';
    const probPercent = (res.probability * 100).toFixed(1);

    resultDiv.innerHTML = `
        <h3><i class="${icon}"></i> ${res.message}</h3>
        <p>Fraud Probability: <strong>${probPercent}%</strong></p>
        <div class="prob-bar">
            <div class="prob-fill" style="width: ${probPercent}%"></div>
        </div>
    `;
    resultDiv.className = className;
    resultDiv.style.display = 'block';
}