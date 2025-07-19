// Modern Desi Debate - Simplified and stable frontend logic

// Global state
let state = {
    initialized: false,
    topic: '',
    currentRound: 0,
    debating: false,
    loading: false
};

// DOM element cache
let elements = {};

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing Desi Debate...');
    
    // Cache DOM elements
    elements = {
        topicInput: document.getElementById('topicInput'),
        topicDisplay: document.getElementById('topicDisplay'),
        currentRound: document.getElementById('currentRound'),
        debateStatus: document.getElementById('debateStatus'),
        debateContent: document.getElementById('debateContent'),
        loadingOverlay: document.getElementById('loadingOverlay'),
        loadingText: document.getElementById('loadingText'),
        startBtn: document.getElementById('startBtn'),
        nextBtn: document.getElementById('nextBtn')
    };
    
    // Bind Enter key
    elements.topicInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            setTopic();
        }
    });
    
    // Initialize system
    initSystem();
});

// Show/hide loading animation (simplified version)
function showLoading(text = 'Processing...') {
    console.log('Show loading:', text);
    elements.loadingText.textContent = text;
    elements.loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    console.log('Hide loading');
    elements.loadingOverlay.style.display = 'none';
}

// Show message
function showMessage(message, type = 'info') {
    console.log(`[${type}] ${message}`);
    
    // Create message element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    alertDiv.style.cssText = 'position: fixed; top: 80px; right: 20px; z-index: 1050; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto remove
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Initialize system
async function initSystem() {
    try {
        showLoading('Initializing system...');
        
        const response = await fetch('/api/init', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        });
        
        const data = await response.json();
        
        if (data.success) {
            state.initialized = true;
            showMessage('System initialized successfully!', 'success');
            updateUI();
        } else {
            throw new Error(data.message || 'Initialization failed');
        }
    } catch (error) {
        console.error('Initialization error:', error);
        showMessage('System initialization failed: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Set topic
async function setTopic() {
    const topic = elements.topicInput.value.trim();
    
    if (!topic) {
        showMessage('Please enter a debate topic', 'warning');
        return;
    }
    
    if (!state.initialized) {
        showMessage('System not initialized yet', 'warning');
        return;
    }
    
    try {
        showLoading('Setting topic...');
        
        const response = await fetch('/api/set_topic', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ topic: topic })
        });
        
        const data = await response.json();
        
        if (data.success) {
            state.topic = topic;
            state.currentRound = 0;
            elements.topicDisplay.textContent = topic;
            showMessage('Topic set successfully!', 'success');
            
            // Clear debate content, show start button
            elements.debateContent.innerHTML = `
                <div class="text-center py-5">
                    <h4>Topic set: ${topic}</h4>
                    <p class="text-muted">Click "Start Debate" button to begin the first round</p>
                </div>
            `;
            
            updateUI();
        } else {
            throw new Error(data.message || 'Setting failed');
        }
    } catch (error) {
        console.error('Set topic error:', error);
        showMessage('Failed to set topic: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Start debate
async function startDebate() {
    if (!state.topic) {
        showMessage('Please set a debate topic first', 'warning');
        return;
    }
    
    state.currentRound = 0;
    elements.debateContent.innerHTML = '';
    await runDebateRound();
}

// Next round
async function nextRound() {
    await runDebateRound();
}

// Execute debate round
async function runDebateRound() {
    if (!state.initialized || !state.topic) {
        showMessage('Please initialize system and set topic first', 'warning');
        return;
    }
    
    if (state.loading) {
        showMessage('Please wait for current operation to complete', 'info');
        return;
    }
    
    try {
        state.loading = true;
        state.debating = true;
        updateUI();
        
        showLoading(state.currentRound === 0 ? 
            'First time loading model, please wait (10-30 seconds)...' : 
            'AI agents are thinking...'
        );
        
        const response = await fetch('/api/debate_round', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                topic: state.topic || 'Default debate topic'
            }),
            // Add timeout control
            signal: AbortSignal.timeout(60000)  // 60 second timeout
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            state.currentRound = data.round;
            elements.currentRound.textContent = data.round;
            
            // Display debate content
            displayDebateRound(data);
            
            // Update Agent states
            if (data.agent_states) {
                updateAgentStates(data.agent_states);
            }
            
            // Check if ended
            if (data.debate_ended) {
                state.debating = false;
                console.log('Debate ended, summary:', data.summary);
                if (data.summary) {
                    showDebateResult(data.summary);
                } else {
                    // If no summary, show basic end message
                    const endDiv = document.createElement('div');
                    endDiv.className = 'debate-result text-center py-5';
                    endDiv.innerHTML = `
                        <h3>Debate Ended</h3>
                        <p class="lead">Debate completed, but unable to generate detailed summary</p>
                    `;
                    elements.debateContent.appendChild(endDiv);
                }
                showMessage('Debate has ended!', 'info');
            } else {
                showMessage(`Round ${data.round} debate completed!`, 'success');
            }
        } else {
            throw new Error(data.message || 'Execution failed');
        }
    } catch (error) {
        console.error('Debate execution error:', error);
        
        // Show different messages based on error type
        if (error.name === 'AbortError') {
            showMessage('Request timeout, please retry', 'error');
        } else if (error.message.includes('fetch')) {
            showMessage('Network connection error, please check connection', 'error');
        } else {
            showMessage('Debate execution failed: ' + error.message, 'error');
        }
        
        // If first round fails, reset debate state
        if (state.currentRound === 0) {
            state.debating = false;
        }
    } finally {
        state.loading = false;
        updateUI();
        hideLoading();
    }
}

// Display debate round
function displayDebateRound(data) {
    // Check data integrity
    if (!data || !data.round) {
        console.error('Debate data incomplete:', data);
        showMessage('Debate data error', 'error');
        return;
    }
    
    const roundDiv = document.createElement('div');
    roundDiv.className = 'debate-round';
    roundDiv.innerHTML = `
        <div class="round-header">
            <div class="round-number">${data.round}</div>
            <h4>Round ${data.round}</h4>
        </div>
    `;
    
    // Check if responses exist and is array
    if (data.responses && Array.isArray(data.responses) && data.responses.length > 0) {
        // Add each AI's response
        data.responses.forEach(response => {
            if (!response || !response.agent_id) {
                console.warn('Skip invalid response:', response);
                return;
            }
            
            const agentType = response.agent_id === 'Agent_A' ? 'support' : 
                             response.agent_id === 'Agent_B' ? 'oppose' : 'neutral';
            const agentName = response.agent_id === 'Agent_A' ? 'Supporter A' :
                             response.agent_id === 'Agent_B' ? 'Opponent B' : 'Neutral C';
            
            const responseDiv = document.createElement('div');
            responseDiv.className = `ai-response ${agentType}`;
            
            // Safely get effect data
            const persuasion = response.effects?.persuasion_score || 0;
            const attack = response.effects?.attack_score || 0;
            
            responseDiv.innerHTML = `
                <div class="response-header">
                    <div class="agent-avatar ${agentType}">
                        <i class="fas ${agentType === 'support' ? 'fa-user-tie' : 
                                       agentType === 'oppose' ? 'fa-user-shield' : 'fa-user-graduate'}"></i>
                    </div>
                    <div>
                        <h5>${agentName}</h5>
                        <small class="text-muted">
                            Persuasion: ${(persuasion * 100).toFixed(0)}% | 
                            Attack: ${(attack * 100).toFixed(0)}%
                        </small>
                    </div>
                </div>
                <div class="response-content">
                    ${response.content || '(No content)'}
                </div>
            `;
            
            roundDiv.appendChild(responseDiv);
        });
    } else {
        // If no response data, show error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-warning';
        errorDiv.textContent = 'No valid response data received for this round';
        roundDiv.appendChild(errorDiv);
    }
    
    elements.debateContent.appendChild(roundDiv);
    
    // Scroll to latest content
    elements.debateContent.scrollTop = elements.debateContent.scrollHeight;
}

// Update Agent states
function updateAgentStates(states) {
    // Check if states exist
    if (!states || typeof states !== 'object') {
        console.warn('Agent state data invalid:', states);
        return;
    }
    
    Object.entries(states).forEach(([agentId, state]) => {
        try {
            const suffix = agentId.split('_')[1];
            
            // Update stance progress bar
            const stanceBar = document.getElementById(`stance${suffix}`);
            if (stanceBar && state.stance !== undefined) {
                const stancePercent = ((state.stance + 1) / 2 * 100).toFixed(0);
                stanceBar.style.width = stancePercent + '%';
                const stanceSpan = stanceBar.querySelector('span');
                if (stanceSpan) {
                    stanceSpan.textContent = 
                        state.stance > 0 ? `+${state.stance.toFixed(2)}` : state.stance.toFixed(2);
                }
            }
            
            // Update conviction progress bar
            const convictionBar = document.getElementById(`conviction${suffix}`);
            if (convictionBar && state.conviction !== undefined) {
                const convictionPercent = (state.conviction * 100).toFixed(0);
                convictionBar.style.width = convictionPercent + '%';
                const convictionSpan = convictionBar.querySelector('span');
                if (convictionSpan) {
                    convictionSpan.textContent = state.conviction.toFixed(2);
                }
            }
            
            // Check surrender status
            if (state.has_surrendered) {
                const agentCard = document.getElementById(`agent${suffix}`);
                if (agentCard) {
                    agentCard.style.opacity = '0.6';
                    showMessage(`${agentId.replace('_', ' ')} has surrendered!`, 'warning');
                }
            }
        } catch (error) {
            console.error(`Error updating ${agentId} state:`, error);
        }
    });
}

// Show debate result
function showDebateResult(summary) {
    // Check if summary exists
    if (!summary) {
        console.error('Debate result data does not exist');
        return;
    }
    
    const resultDiv = document.createElement('div');
    resultDiv.className = 'debate-result text-center py-5';
    
    // Safely build result HTML
    let scoresHtml = '';
    if (summary.scores && typeof summary.scores === 'object') {
        scoresHtml = Object.entries(summary.scores)
            .sort(([,a], [,b]) => b - a)
            .map(([agent, score]) => `
                <div class="mb-2">
                    ${agent}: ${(score || 0).toFixed(1)} points
                    ${agent === summary.winner ? '<span class="badge bg-warning ms-2">Winner</span>' : ''}
                </div>
            `).join('');
    }
    
    resultDiv.innerHTML = `
        <h3>Debate Ended</h3>
        <p class="lead">${summary.verdict || 'Debate completed'}</p>
        <div class="mt-4">
            <h5>Final Scores</h5>
            ${scoresHtml || '<p class="text-muted">No scoring data</p>'}
        </div>
    `;
    
    elements.debateContent.appendChild(resultDiv);
}

// Reset debate
async function resetDebate() {
    if (state.loading) {
        showMessage('Please wait for current operation to complete', 'info');
        return;
    }
    
    if (!confirm('Are you sure you want to reset the debate?')) {
        return;
    }
    
    try {
        showLoading('Resetting...');
        
        const response = await fetch('/api/reset', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Reset state - keep initialized as true
            state.topic = '';
            state.currentRound = 0;
            state.debating = false;
            state.loading = false;  // Ensure loading state is reset
            
            // Reset UI
            elements.topicInput.value = '';
            elements.topicDisplay.textContent = 'Please set a debate topic to begin';
            elements.currentRound.textContent = '0';
            elements.debateContent.innerHTML = `
                <div class="welcome-screen">
                    <div class="welcome-icon">
                        <i class="fas fa-robot"></i>
                    </div>
                    <h3>Welcome to Desi Debate</h3>
                    <p>This is a multi-agent debate system where three AI agents discuss your topic from different perspectives</p>
                </div>
            `;
            
            // Reset progress bars
            resetAgentStates();
            
            showMessage('Debate reset successfully', 'success');
            updateUI();
        } else {
            throw new Error(data.message || 'Reset failed');
        }
    } catch (error) {
        console.error('Reset error:', error);
        showMessage('Reset failed: ' + error.message, 'error');
        // Reset loading state even if failed
        state.loading = false;
        updateUI();
    } finally {
        hideLoading();
    }
}

// Reset Agent states
function resetAgentStates() {
    // Agent A
    document.getElementById('stanceA').style.width = '80%';
    document.getElementById('stanceA').querySelector('span').textContent = '+0.8';
    document.getElementById('convictionA').style.width = '70%';
    document.getElementById('convictionA').querySelector('span').textContent = '0.7';
    
    // Agent B
    document.getElementById('stanceB').style.width = '30%';
    document.getElementById('stanceB').querySelector('span').textContent = '-0.6';
    document.getElementById('convictionB').style.width = '60%';
    document.getElementById('convictionB').querySelector('span').textContent = '0.6';
    
    // Agent C
    document.getElementById('stanceC').style.width = '50%';
    document.getElementById('stanceC').querySelector('span').textContent = '0.0';
    document.getElementById('convictionC').style.width = '50%';
    document.getElementById('convictionC').querySelector('span').textContent = '0.5';
    
    // Remove surrender state
    document.querySelectorAll('.agent-card').forEach(card => {
        card.style.opacity = '1';
    });
}

// Update UI state
function updateUI() {
    // Update button states
    elements.startBtn.disabled = !state.initialized || !state.topic || state.loading || state.debating;
    elements.nextBtn.disabled = !state.initialized || !state.topic || state.loading || !state.debating || state.currentRound === 0;
    
    // Update status display
    if (state.debating) {
        elements.debateStatus.textContent = 'In Progress';
        elements.debateStatus.style.color = 'var(--success-color)';
    } else if (state.topic) {
        elements.debateStatus.textContent = 'Ready';
        elements.debateStatus.style.color = 'var(--info-color)';
    } else {
        elements.debateStatus.textContent = 'Waiting Setup';
        elements.debateStatus.style.color = 'var(--warning-color)';
    }
}

// Export debate records
async function exportDebate() {
    if (state.currentRound === 0) {
        showMessage('No debate records to export', 'warning');
        return;
    }
    
    try {
        showLoading('Exporting...');
        
        const response = await fetch('/api/export');
        const data = await response.json();
        
        if (data.success) {
            // Create download
            const blob = new Blob([JSON.stringify(data.data, null, 2)], 
                                 { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `debate_${new Date().toISOString().slice(0, 10)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            showMessage('Debate records exported successfully', 'success');
        } else {
            throw new Error(data.message || 'Export failed');
        }
    } catch (error) {
        console.error('Export error:', error);
        showMessage('Export failed: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Show statistics
function showStats() {
    showMessage('Statistics feature under development...', 'info');
}

// Show about
function showAbout() {
    showMessage('Desi Debate - Multi-agent Debate System v1.0', 'info');
}

// Toggle theme
function toggleTheme() {
    document.body.classList.toggle('dark-theme');
    const icon = document.getElementById('themeIcon');
    icon.className = document.body.classList.contains('dark-theme') ? 
                     'fas fa-sun' : 'fas fa-moon';
} 