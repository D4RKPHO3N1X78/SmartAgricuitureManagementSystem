document.addEventListener('DOMContentLoaded', () => {
    // Prediction Elements
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const telemetryInfo = document.getElementById('telemetry-info');
    const latencyVal = document.getElementById('latency-val');
    
    // Tab Elements
    const predictTab = document.getElementById('nav-predict');
    const analyticsTab = document.getElementById('nav-analytics');
    const predictorView = document.getElementById('predictor-view');
    const analyticsView = document.getElementById('analytics-view');
    
    // Stat Elements
    const statTotal = document.getElementById('stat-total');
    const statAvg = document.getElementById('stat-avg');
    
    // Chart References
    let trendChartInstance = null;
    let scatterChartInstance = null;
    
    // Create loader element
    const loader = document.createElement('div');
    loader.className = 'loader-spinner';
    
    // Tab Switching Logic
    predictTab.addEventListener('click', () => {
        predictTab.classList.add('active');
        analyticsTab.classList.remove('active');
        predictorView.style.display = 'grid'; // Grid is used in CSS
        analyticsView.style.display = 'none';
        
        // Trigger resize on window for responsive layout fixes
        window.dispatchEvent(new Event('resize'));
    });
    
    analyticsTab.addEventListener('click', async () => {
        analyticsTab.classList.add('active');
        predictTab.classList.remove('active');
        predictorView.style.display = 'none';
        analyticsView.style.display = 'flex'; // Flex column in CSS
        
        await loadAnalytics();
    });

    // Prediction Form Logic
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        resultContainer.innerHTML = '';
        resultContainer.appendChild(loader);
        telemetryInfo.style.display = 'none';
        
        const payload = {
            temperature: parseFloat(document.getElementById('temp').value),
            humidity: parseFloat(document.getElementById('hum').value),
            soil_ph: parseFloat(document.getElementById('ph').value),
            rainfall: parseFloat(document.getElementById('rain').value),
            nitrogen: parseFloat(document.getElementById('nitro').value),
            phosphorus: parseFloat(document.getElementById('phos').value),
            potassium: parseFloat(document.getElementById('pota').value)
        };
        
        try {
            const start = Date.now();
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const data = await response.json();
            
            const elapsed = Date.now() - start;
            if (elapsed < 600) {
                await new Promise(resolve => setTimeout(resolve, 600 - elapsed));
            }
            
            resultContainer.innerHTML = `
                <div class="success-result">
                    <div class="yield-value">${data.prediction_tonnes_ha.toFixed(2)}</div>
                    <div class="yield-label">Tonnes per Hectare</div>
                </div>
            `;
            
            latencyVal.textContent = data.inference_time_ms;
            telemetryInfo.style.display = 'flex';
            
        } catch (error) {
            console.error('Error during prediction:', error);
            resultContainer.innerHTML = `
                <div class="waiting-state" style="color: #ef4444;">
                    <i class="fa-solid fa-triangle-exclamation"></i>
                    <p>Inference Failed</p>
                    <small style="opacity: 0.7; margin-top: 10px; display: block;">Check console logs</small>
                </div>
            `;
        }
    });

    // Analytics Fetch Logic
    async function loadAnalytics() {
        try {
            const response = await fetch('/analytics');
            if (!response.ok) throw new Error("Failed to load analytics");
            
            const data = await response.json();
            
            // Update Stat Cards
            statTotal.textContent = data.summary.totalInferences;
            statAvg.innerHTML = `${data.summary.avgYield.toFixed(2)} <span style="font-size: 14px;">t/ha</span>`;
            
            // Prepare Chart Data
            const labelsWindow = data.history.map((_, i) => `Infer ${i+1}`);
            const yieldOverTime = data.history.map(item => item.yield_pred);
            const scatterData = data.history.map(item => ({
                x: item.inputs.temperature,
                y: item.yield_pred
            }));
            
            updateTrendChart(labelsWindow, yieldOverTime);
            updateScatterChart(scatterData);
            
        } catch (err) {
            console.error("Analytics Error: ", err);
        }
    }

    // Chart.js Generators
    function updateTrendChart(labels, yields) {
        const ctx = document.getElementById('trendChart').getContext('2d');
        
        // Destroy existing instance to prevent overlapping issues when re-toggling
        if(trendChartInstance) trendChartInstance.destroy();
        
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = 'Inter';
        
        trendChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Predicted Yield (t/ha)',
                    data: yields,
                    borderColor: '#00ffa3',
                    backgroundColor: 'rgba(0, 255, 163, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#00b8ff',
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { 
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        beginAtZero: false 
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    function updateScatterChart(scatterData) {
        const ctx = document.getElementById('scatterChart').getContext('2d');
        
        if(scatterChartInstance) scatterChartInstance.destroy();
        
        scatterChartInstance = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Temp vs Yield',
                    data: scatterData,
                    backgroundColor: '#00b8ff',
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Temperature (°C)' },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    },
                    y: {
                        title: { display: true, text: 'Predicted Yield (t/ha)' },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    }
                },
                plugins: { legend: { display: false } }
            }
        });
    }
});
