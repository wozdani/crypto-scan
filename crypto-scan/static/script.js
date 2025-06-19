// Global variables
let refreshInterval;
let isRefreshing = false;

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    startAutoRefresh();
});

function initializeDashboard() {
    console.log('🚀 Initializing Crypto Scanner Dashboard');
    
    // Load initial data
    refreshSystemStatus();
    refreshMarketOverview();
    refreshTopPerformers();
    refreshRecentAlerts();
    refreshGptAnalyses();
    
    // Update timestamp
    updateLastUpdateTime();
}

function startAutoRefresh() {
    // Refresh data every 30 seconds
    refreshInterval = setInterval(() => {
        if (!isRefreshing) {
            refreshSystemStatus();
            refreshMarketOverview();
            refreshTopPerformers();
            refreshRecentAlerts();
            updateLastUpdateTime();
        }
    }, 30000);
    
    console.log('🔄 Auto-refresh started (30s interval)');
}

function stopAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
        console.log('⏹️ Auto-refresh stopped');
    }
}

function updateLastUpdateTime() {
    const lastUpdate = document.getElementById('last-update');
    if (lastUpdate) {
        lastUpdate.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
    }
}

// System Status Functions
async function refreshSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.error) {
            updateStatusIndicator('error', 'Error');
            return;
        }
        
        updateStatusIndicator('online', 'Online');
        updateConnectionStatus(data.connections);
        updateDataStatus(data.data_stats);
        
    } catch (error) {
        console.error('Error fetching system status:', error);
        updateStatusIndicator('error', 'Error');
    }
}

function updateStatusIndicator(status, text) {
    const indicator = document.getElementById('status-indicator');
    if (!indicator) return;
    
    // Remove existing classes
    indicator.className = 'badge me-3';
    
    // Add status-specific class and icon
    switch (status) {
        case 'online':
            indicator.classList.add('bg-success');
            indicator.innerHTML = '<i class="fas fa-circle"></i> ' + text;
            break;
        case 'warning':
            indicator.classList.add('bg-warning');
            indicator.innerHTML = '<i class="fas fa-exclamation-triangle"></i> ' + text;
            break;
        case 'error':
            indicator.classList.add('bg-danger');
            indicator.innerHTML = '<i class="fas fa-times-circle"></i> ' + text;
            break;
        default:
            indicator.classList.add('bg-secondary');
            indicator.innerHTML = '<i class="fas fa-circle"></i> ' + text;
    }
}

function updateConnectionStatus(connections) {
    // Update Telegram status
    const telegramStatus = document.getElementById('telegram-status');
    if (telegramStatus && connections.telegram) {
        const telegram = connections.telegram;
        if (telegram.status === 'ok') {
            telegramStatus.innerHTML = `<span class="text-success"><i class="fas fa-check"></i> ${telegram.message}</span>`;
        } else {
            telegramStatus.innerHTML = `<span class="text-danger"><i class="fas fa-times"></i> ${telegram.message}</span>`;
        }
    }
    
    // Update OpenAI status
    const openaiStatus = document.getElementById('openai-status');
    if (openaiStatus && connections.openai) {
        const openai = connections.openai;
        if (openai.status === 'ok') {
            openaiStatus.innerHTML = `<span class="text-success"><i class="fas fa-check"></i> ${openai.message}</span>`;
        } else {
            openaiStatus.innerHTML = `<span class="text-danger"><i class="fas fa-times"></i> ${openai.message}</span>`;
        }
    }
}

function updateDataStatus(dataStats) {
    const dataStatus = document.getElementById('data-status');
    if (!dataStatus || !dataStats) return;
    
    const alerts = dataStats.alerts;
    const scores = dataStats.scores;
    
    if (alerts && alerts.exists && scores && scores.exists) {
        dataStatus.innerHTML = `<span class="text-success"><i class="fas fa-check"></i> ${alerts.record_count + scores.record_count} records</span>`;
    } else {
        dataStatus.innerHTML = `<span class="text-warning"><i class="fas fa-exclamation-triangle"></i> Missing data files</span>`;
    }
}

// Market Overview Functions
async function refreshMarketOverview() {
    try {
        const response = await fetch('/api/market-overview');
        const data = await response.json();
        
        if (data.error) {
            console.error('Error fetching market overview:', data.error);
            return;
        }
        
        const overview = data.overview;
        
        // Update metrics
        updateElement('alerts-24h', overview.alert_count_24h || 0);
        updateElement('avg-score-24h', overview.avg_score_24h || 0);
        updateElement('high-scores-24h', overview.high_score_count_24h || 0);
        
        // Update top performer
        if (overview.top_performers && overview.top_performers.length > 0) {
            updateElement('top-symbol', overview.top_performers[0].symbol);
        } else {
            updateElement('top-symbol', 'N/A');
        }
        
    } catch (error) {
        console.error('Error fetching market overview:', error);
    }
}

// Top Performers Functions
async function refreshTopPerformers() {
    const container = document.getElementById('top-performers-list');
    if (!container) return;
    
    try {
        showLoading(container);
        
        const response = await fetch('/api/top-performers?hours=24&limit=10');
        const data = await response.json();
        
        if (data.error) {
            showError(container, data.error);
            return;
        }
        
        if (!data.performers || data.performers.length === 0) {
            showNoData(container, 'No performers in the last 24 hours');
            return;
        }
        
        container.innerHTML = data.performers.map(performer => createPerformerItem(performer)).join('');
        
    } catch (error) {
        console.error('Error fetching top performers:', error);
        showError(container, 'Failed to load top performers');
    }
}

function createPerformerItem(performer) {
    const scoreClass = getScoreClass(performer.score);
    const timestamp = new Date(performer.timestamp).toLocaleTimeString();
    
    return `
        <div class="performer-item">
            <div class="d-flex align-items-center">
                <span class="symbol-link me-3" onclick="showSymbolDetails('${performer.symbol}')">
                    ${performer.symbol}
                </span>
                <span class="timestamp">${timestamp}</span>
            </div>
            <span class="score-badge ${scoreClass}">
                ${performer.score}
            </span>
        </div>
    `;
}

// Whale Priority Functions
async function refreshWhalePriority() {
    const container = document.getElementById('whale-priority-list');
    if (!container) return;
    
    try {
        showLoading(container);
        
        const response = await fetch('/api/whale-priority');
        const data = await response.json();
        
        if (data.error) {
            showError(container, data.error);
            return;
        }
        
        if (!data.priority_tokens || data.priority_tokens.length === 0) {
            showNoData(container, 'No whale priority tokens detected');
            return;
        }
        
        container.innerHTML = data.priority_tokens.slice(0, 8).map(token => createWhalePriorityItem(token)).join('');
        
    } catch (error) {
        console.error('Error fetching whale priority:', error);
        showError(container, 'Failed to load whale priority data');
    }
}

function createWhalePriorityItem(token) {
    const priorityClass = getPriorityClass(token.priority_score);
    const patternIcon = getWhalePatternIcon(token.whale_pattern);
    const underWatchBadge = token.under_watch ? '<span class="badge bg-warning ms-2">WATCH</span>' : '';
    
    return `
        <div class="whale-priority-item mb-2 p-2 border rounded">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <span class="symbol-link fw-bold" onclick="showSymbolDetails('${token.symbol}')">
                        ${patternIcon} ${token.symbol}
                    </span>
                    ${underWatchBadge}
                    <div class="small text-muted">
                        ${token.whale_count} whale TX, ${token.minutes_ago}min ago
                    </div>
                </div>
                <div class="text-end">
                    <span class="priority-score ${priorityClass}">
                        ${token.priority_score}
                    </span>
                    ${token.whale_score_boost > 0 ? `<div class="small text-success">+${token.whale_score_boost} boost</div>` : ''}
                </div>
            </div>
        </div>
    `;
}

function getPriorityClass(score) {
    if (score >= 80) return 'text-danger fw-bold';
    if (score >= 60) return 'text-warning fw-bold';
    if (score >= 40) return 'text-info';
    return 'text-muted';
}

function getWhalePatternIcon(pattern) {
    switch (pattern) {
        case 'repeat_cluster': return '🔄';
        case 'double_whale': return '🐋🐋';
        case 'repeat_address_cluster': return '📍';
        default: return '🐋';
    }
}

// Recent Alerts Functions
async function refreshRecentAlerts() {
    const container = document.getElementById('recent-alerts-list');
    if (!container) return;
    
    try {
        showLoading(container);
        
        const response = await fetch('/api/recent-alerts?hours=24');
        const data = await response.json();
        
        if (data.error) {
            showError(container, data.error);
            return;
        }
        
        if (!data.alerts || data.alerts.length === 0) {
            showNoData(container, 'No alerts in the last 24 hours');
            return;
        }
        
        container.innerHTML = data.alerts.slice(0, 10).map(alert => createAlertItem(alert)).join('');
        
    } catch (error) {
        console.error('Error fetching recent alerts:', error);
        showError(container, 'Failed to load recent alerts');
    }
}

function createAlertItem(alert) {
    const timestamp = new Date(alert.timestamp).toLocaleTimeString();
    const message = alert.message || 'No message';
    
    // Extract score from message
    let scoreClass = '';
    const scoreMatch = message.match(/PPWCS:\s*(\d+\.?\d*)/);
    if (scoreMatch) {
        const score = parseFloat(scoreMatch[1]);
        scoreClass = getScoreClass(score);
    }
    
    return `
        <div class="alert-item ${scoreClass ? 'alert-' + scoreClass.split('-')[1] + '-score' : ''}">
            <div class="d-flex justify-content-between align-items-start">
                <div class="flex-grow-1">
                    <div class="fw-bold mb-1">${extractSymbolFromMessage(message) || 'System'}</div>
                    <div class="small text-muted">${message.replace(/\*/g, '')}</div>
                </div>
                <div class="text-end">
                    <div class="timestamp">${timestamp}</div>
                    ${alert.status === 'success' ? '<i class="fas fa-check text-success"></i>' : 
                      alert.status === 'failed' ? '<i class="fas fa-times text-danger"></i>' : ''}
                </div>
            </div>
        </div>
    `;
}

// GPT Analyses Functions
async function refreshGptAnalyses() {
    const container = document.getElementById('gpt-analyses-list');
    if (!container) return;
    
    try {
        showLoading(container);
        
        const response = await fetch('/api/gpt-analyses?hours=24&limit=5');
        const data = await response.json();
        
        if (data.error) {
            showError(container, data.error);
            return;
        }
        
        if (!data.analyses || data.analyses.length === 0) {
            showNoData(container, 'No GPT analyses in the last 24 hours');
            return;
        }
        
        container.innerHTML = data.analyses.map(analysis => createGptAnalysisItem(analysis)).join('');
        
    } catch (error) {
        console.error('Error fetching GPT analyses:', error);
        showError(container, 'Failed to load GPT analyses');
    }
}

function createGptAnalysisItem(analysis) {
    const timestamp = new Date(analysis.timestamp).toLocaleString();
    const gptAnalysis = analysis.analysis;
    
    if (!gptAnalysis) {
        return `
            <div class="gpt-analysis">
                <div class="text-muted">Invalid analysis data</div>
            </div>
        `;
    }
    
    const confidenceClass = getConfidenceClass(gptAnalysis.confidence_level || 0);
    
    return `
        <div class="gpt-analysis">
            <div class="gpt-analysis-header">
                <div class="d-flex align-items-center">
                    <span class="symbol-link me-3" onclick="showSymbolDetails('${analysis.symbol}')">
                        ${analysis.symbol}
                    </span>
                    <span class="score-badge ${getScoreClass(analysis.score)}">${analysis.score}</span>
                </div>
                <div class="timestamp">${timestamp}</div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-2">
                        <strong>Risk Assessment:</strong> 
                        <span class="badge bg-${getRiskBadgeClass(gptAnalysis.risk_assessment)}">${gptAnalysis.risk_assessment || 'Unknown'}</span>
                    </div>
                    <div class="mb-2">
                        <strong>Confidence:</strong> ${gptAnalysis.confidence_level || 0}%
                        <div class="confidence-bar">
                            <div class="confidence-fill ${confidenceClass}" style="width: ${gptAnalysis.confidence_level || 0}%"></div>
                        </div>
                    </div>
                    <div class="mb-2">
                        <strong>Prediction:</strong> 
                        <span class="badge bg-${getPredictionBadgeClass(gptAnalysis.price_prediction)}">${gptAnalysis.price_prediction || 'Unknown'}</span>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="mb-2">
                        <strong>Entry Recommendation:</strong> 
                        <span class="badge bg-${getEntryBadgeClass(gptAnalysis.entry_recommendation)}">${gptAnalysis.entry_recommendation || 'Unknown'}</span>
                    </div>
                    <div class="mb-2">
                        <strong>Time Horizon:</strong> ${gptAnalysis.time_horizon || 'Unknown'}
                    </div>
                </div>
            </div>
            
            ${gptAnalysis.summary ? `
                <div class="mt-3 p-2 bg-light rounded">
                    <strong>Summary:</strong> ${gptAnalysis.summary}
                </div>
            ` : ''}
            
            ${gptAnalysis.key_indicators && gptAnalysis.key_indicators.length > 0 ? `
                <div class="mt-2">
                    <strong>Key Indicators:</strong>
                    ${gptAnalysis.key_indicators.map(indicator => `<span class="badge bg-info me-1">${indicator}</span>`).join('')}
                </div>
            ` : ''}
        </div>
    `;
}

// Symbol Details Modal
async function showSymbolDetails(symbol) {
    const modal = new bootstrap.Modal(document.getElementById('symbolModal'));
    const modalLabel = document.getElementById('symbolModalLabel');
    const modalBody = document.getElementById('symbolModalBody');
    
    modalLabel.textContent = `${symbol} - Symbol Details`;
    modalBody.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';
    
    modal.show();
    
    try {
        const response = await fetch(`/api/symbol/${symbol}`);
        const data = await response.json();
        
        if (data.error) {
            modalBody.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            return;
        }
        
        modalBody.innerHTML = createSymbolDetailsContent(data);
        
    } catch (error) {
        console.error('Error fetching symbol details:', error);
        modalBody.innerHTML = '<div class="alert alert-danger">Failed to load symbol details</div>';
    }
}

function createSymbolDetailsContent(data) {
    const stats = data.stats;
    const recentScores = data.recent_scores || [];
    
    let content = `
        <div class="row">
            <div class="col-md-6">
                <h6>Statistics (7 days)</h6>
    `;
    
    if (stats) {
        content += `
                <div class="mb-2"><strong>Scan Count:</strong> ${stats.count}</div>
                <div class="mb-2"><strong>Average Score:</strong> ${stats.avg_score.toFixed(1)}</div>
                <div class="mb-2"><strong>Max Score:</strong> ${stats.max_score}</div>
                <div class="mb-2"><strong>Min Score:</strong> ${stats.min_score}</div>
                <div class="mb-2"><strong>Latest Score:</strong> ${stats.latest_score}</div>
        `;
    } else {
        content += '<div class="text-muted">No statistics available</div>';
    }
    
    content += `
            </div>
            <div class="col-md-6">
                <h6>Recent Activity</h6>
                <div style="max-height: 300px; overflow-y: auto;">
    `;
    
    if (recentScores.length > 0) {
        content += recentScores.slice(-10).reverse().map(score => {
            const timestamp = new Date(score.timestamp).toLocaleString();
            const scoreClass = getScoreClass(score.score);
            return `
                <div class="d-flex justify-content-between align-items-center mb-2 p-2 bg-light rounded">
                    <span class="small">${timestamp}</span>
                    <span class="score-badge ${scoreClass}">${score.score}</span>
                </div>
            `;
        }).join('');
    } else {
        content += '<div class="text-muted">No recent activity</div>';
    }
    
    content += `
                </div>
            </div>
        </div>
    `;
    
    return content;
}

// Utility Functions
function getScoreClass(score) {
    if (score >= 80) return 'score-high';
    if (score >= 60) return 'score-medium';
    return 'score-low';
}

function getConfidenceClass(confidence) {
    if (confidence >= 80) return 'confidence-high';
    if (confidence >= 50) return 'confidence-medium';
    return 'confidence-low';
}

function getRiskBadgeClass(risk) {
    switch (risk) {
        case 'low': return 'success';
        case 'medium': return 'warning';
        case 'high': return 'danger';
        default: return 'secondary';
    }
}

function getPredictionBadgeClass(prediction) {
    switch (prediction) {
        case 'bullish': return 'success';
        case 'bearish': return 'danger';
        case 'neutral': return 'secondary';
        default: return 'secondary';
    }
}

function getEntryBadgeClass(entry) {
    switch (entry) {
        case 'immediate': return 'success';
        case 'wait': return 'warning';
        case 'avoid': return 'danger';
        default: return 'secondary';
    }
}

function extractSymbolFromMessage(message) {
    const match = message.match(/\*([A-Z0-9]+)\*/);
    return match ? match[1] : null;
}

function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function showLoading(container) {
    container.innerHTML = '<div class="text-center text-muted loading-pulse"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';
}

function showError(container, message) {
    container.innerHTML = `<div class="alert alert-danger"><i class="fas fa-exclamation-triangle"></i> ${message}</div>`;
}

function showNoData(container, message) {
    container.innerHTML = `<div class="no-data"><i class="fas fa-inbox"></i><br>${message}</div>`;
}

// Manual refresh functions
function refreshAll() {
    if (isRefreshing) return;
    
    isRefreshing = true;
    console.log('🔄 Manual refresh initiated');
    
    Promise.all([
        refreshSystemStatus(),
        refreshMarketOverview(),
        refreshTopPerformers(),
        refreshWhalePriority(),
        refreshRecentAlerts(),
        refreshGptAnalyses()
    ]).finally(() => {
        isRefreshing = false;
        updateLastUpdateTime();
        console.log('✅ Manual refresh completed');
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl+R or F5 for refresh
    if ((event.ctrlKey && event.key === 'r') || event.key === 'F5') {
        event.preventDefault();
        refreshAll();
    }
    
    // Escape to close modals
    if (event.key === 'Escape') {
        const openModals = document.querySelectorAll('.modal.show');
        openModals.forEach(modal => {
            const modalInstance = bootstrap.Modal.getInstance(modal);
            if (modalInstance) {
                modalInstance.hide();
            }
        });
    }
});

// Page visibility handling
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        stopAutoRefresh();
    } else {
        startAutoRefresh();
        refreshAll();
    }
});

console.log('📊 Crypto Scanner Dashboard script loaded');
