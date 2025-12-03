// Bloomberg-style Console JavaScript

// Update time display
function updateTime() {
    const now = new Date();
    const timeString = now.toTimeString().split(' ')[0];
    document.getElementById('time').textContent = timeString;
}

// Update time every second
setInterval(updateTime, 1000);
updateTime();

// Console output handler
const consoleOutput = document.getElementById('consoleOutput');
const consoleInput = document.getElementById('consoleInput');

// Simulate live data updates
function simulateDataUpdate() {
    const dataRows = document.querySelectorAll('.data-row');
    dataRows.forEach(row => {
        const priceElement = row.querySelector('.price');
        const changeElement = row.querySelector('.change');
        
        if (priceElement && changeElement) {
            const currentPrice = parseFloat(priceElement.textContent.replace(/,/g, ''));
            const randomChange = (Math.random() - 0.5) * 0.5; // Small random change percentage
            const newPrice = (currentPrice * (1 + randomChange / 100)).toFixed(2);
            const changePercent = randomChange.toFixed(2);
            
            priceElement.textContent = parseFloat(newPrice).toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            });
            
            changeElement.textContent = Math.abs(changePercent) + '%';
            
            // Update classes based on change
            if (changePercent > 0) {
                priceElement.classList.remove('negative');
                priceElement.classList.add('positive');
                changeElement.classList.remove('negative');
                changeElement.classList.add('positive');
            } else if (changePercent < 0) {
                priceElement.classList.remove('positive');
                priceElement.classList.add('negative');
                changeElement.classList.remove('positive');
                changeElement.classList.add('negative');
            }
        }
    });
}

// Update data every 5 seconds
setInterval(simulateDataUpdate, 5000);

// Command history
const commandHistory = [];
let historyIndex = -1;

// Available commands
const commands = {
    help: () => {
        return `
<span class="info">Available Commands:</span>
  help      - Display this help message
  clear     - Clear the console output
  status    - Show system status
  time      - Display current time
  market    - Show market summary
  about     - About this terminal
  quote [ticker] - Get quote for ticker
        `.trim();
    },
    clear: () => {
        consoleOutput.innerHTML = '';
        return null;
    },
    status: () => {
        return `<span class="success">✓ System Status: OPERATIONAL</span>
<span class="success">✓ Market Feed: CONNECTED</span>
<span class="success">✓ Data Stream: ACTIVE</span>
<span class="info">  Uptime: ${Math.floor(Math.random() * 24)}h ${Math.floor(Math.random() * 60)}m</span>`;
    },
    time: () => {
        const now = new Date();
        return `<span class="info">Current Time: ${now.toLocaleString()}</span>`;
    },
    market: () => {
        return `<span class="info">Market Summary:</span>
  SPX:    4,783.45  <span class="change positive">+1.24%</span>
  NASDAQ: 15,011.35 <span class="change positive">+0.89%</span>
  DOW:    37,305.16 <span class="change negative">-0.45%</span>`;
    },
    about: () => {
        return `<span class="info">WAVES SIMPLE CONSOLE</span>
<span class="info">Bloomberg-Style Terminal Interface</span>
<span class="info">Version 1.0.0</span>
<span class="info">© 2025 Waves Terminal</span>`;
    },
    quote: (args) => {
        const ticker = args[0]?.toUpperCase() || 'UNKNOWN';
        const price = (Math.random() * 1000 + 100).toFixed(2);
        const change = (Math.random() * 10 - 5).toFixed(2);
        const changeClass = change >= 0 ? 'positive' : 'negative';
        return `<span class="info">Quote for ${ticker}:</span>
  Price:  $${price}
  Change: <span class="change ${changeClass}">${change > 0 ? '+' : ''}${change}%</span>`;
    }
};

// Add line to console
function addConsoleLine(text, type = '') {
    const line = document.createElement('div');
    line.className = `console-line ${type}`;
    line.innerHTML = text;
    consoleOutput.appendChild(line);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

// Process command
function processCommand(input) {
    const trimmedInput = input.trim();
    
    if (!trimmedInput) return;
    
    // Add command to history
    commandHistory.unshift(trimmedInput);
    historyIndex = -1;
    
    // Display the command
    addConsoleLine(`<span class="prompt">WAVES&gt;</span> ${trimmedInput}`);
    
    // Parse command and arguments
    const parts = trimmedInput.toLowerCase().split(' ');
    const command = parts[0];
    const args = parts.slice(1);
    
    // Execute command
    if (commands[command]) {
        const result = commands[command](args);
        if (result !== null) {
            addConsoleLine(result);
        }
    } else {
        addConsoleLine(`<span class="error">ERROR: Unknown command '${command}'. Type 'help' for available commands.</span>`, 'error');
    }
}

// Handle input
consoleInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        const input = consoleInput.value;
        processCommand(input);
        consoleInput.value = '';
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (historyIndex < commandHistory.length - 1) {
            historyIndex++;
            consoleInput.value = commandHistory[historyIndex];
        }
    } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (historyIndex > 0) {
            historyIndex--;
            consoleInput.value = commandHistory[historyIndex];
        } else if (historyIndex === 0) {
            historyIndex = -1;
            consoleInput.value = '';
        }
    }
});

// Auto-focus input
consoleInput.focus();
document.addEventListener('click', () => {
    consoleInput.focus();
});

// Initial welcome message
setTimeout(() => {
    addConsoleLine('<span class="info">Type "help" to see available commands</span>', 'info');
}, 500);

// Simulate occasional system messages
function randomSystemMessage() {
    const messages = [
        '<span class="info">INFO: Market data refreshed</span>',
        '<span class="success">SUCCESS: Connection stable</span>',
        '<span class="info">INFO: Processing real-time updates</span>',
        '<span class="info">INFO: Data stream optimal</span>'
    ];
    
    const randomMessage = messages[Math.floor(Math.random() * messages.length)];
    addConsoleLine(randomMessage);
}

// Add random messages occasionally (every 30-60 seconds)
setInterval(() => {
    if (Math.random() > 0.5) {
        randomSystemMessage();
    }
}, 45000);

console.log('Bloomberg-style console initialized');
