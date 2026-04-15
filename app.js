/* =====================================================
   Q-learning vs SARSA — Cliff Walking Implementation
   ===================================================== */

// ── Environment Constants ──
const ROWS = 4;
const COLS = 12;
const START = [3, 0];       // bottom-left
const GOAL = [3, 11];       // bottom-right
const ACTIONS = [
    [-1, 0], // up
    [1, 0],  // down
    [0, -1], // left
    [0, 1]   // right
];
const ACTION_ARROWS = ['↑', '↓', '←', '→'];

// ── Cliff cells: row 3, col 1~10 ──
function isCliff(r, c) {
    return r === 3 && c >= 1 && c <= 10;
}

function isGoal(r, c) {
    return r === GOAL[0] && c === GOAL[1];
}

function isStart(r, c) {
    return r === START[0] && c === START[1];
}

// ── Take a step in the environment ──
function step(r, c, actionIdx) {
    let nr = r + ACTIONS[actionIdx][0];
    let nc = c + ACTIONS[actionIdx][1];

    // Boundary check
    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) {
        nr = r;
        nc = c;
    }

    // Cliff → reward = -100, return to start
    if (isCliff(nr, nc)) {
        return { nr: START[0], nc: START[1], reward: -100, done: false, fellOff: true };
    }

    // Goal → done
    if (isGoal(nr, nc)) {
        return { nr, nc, reward: -1, done: true, fellOff: false };
    }

    return { nr, nc, reward: -1, done: false, fellOff: false };
}

// ── Initialize Q-table ──
function initQ() {
    const Q = [];
    for (let r = 0; r < ROWS; r++) {
        Q[r] = [];
        for (let c = 0; c < COLS; c++) {
            Q[r][c] = new Float64Array(4); // [up, down, left, right] = 0
        }
    }
    return Q;
}

// ── ε-greedy action selection ──
function epsilonGreedy(Q, r, c, epsilon) {
    if (Math.random() < epsilon) {
        return Math.floor(Math.random() * 4);
    }
    let bestA = 0;
    let bestVal = Q[r][c][0];
    for (let a = 1; a < 4; a++) {
        if (Q[r][c][a] > bestVal) {
            bestVal = Q[r][c][a];
            bestA = a;
        }
    }
    return bestA;
}

// ── Greedy action (for policy extraction) ──
function greedyAction(Q, r, c) {
    let bestA = 0;
    let bestVal = Q[r][c][0];
    for (let a = 1; a < 4; a++) {
        if (Q[r][c][a] > bestVal) {
            bestVal = Q[r][c][a];
            bestA = a;
        }
    }
    return bestA;
}

// ── Q-learning training ──
function trainQLearning(episodes, alpha, gamma, epsilon) {
    const Q = initQ();
    const rewards = [];
    let totalCliffFalls = 0;

    for (let ep = 0; ep < episodes; ep++) {
        let r = START[0], c = START[1];
        let totalReward = 0;
        let steps = 0;
        const maxSteps = 10000;

        while (!isGoal(r, c) && steps < maxSteps) {
            const a = epsilonGreedy(Q, r, c, epsilon);
            const result = step(r, c, a);
            if (result.fellOff) totalCliffFalls++;

            // Q-learning update: use max over next actions
            let maxNextQ = Q[result.nr][result.nc][0];
            for (let a2 = 1; a2 < 4; a2++) {
                if (Q[result.nr][result.nc][a2] > maxNextQ) {
                    maxNextQ = Q[result.nr][result.nc][a2];
                }
            }

            Q[r][c][a] += alpha * (result.reward + gamma * maxNextQ - Q[r][c][a]);

            totalReward += result.reward;
            r = result.nr;
            c = result.nc;
            steps++;
        }
        rewards.push(totalReward);
    }

    return { Q, rewards, cliffFalls: totalCliffFalls };
}

// ── SARSA training ──
function trainSARSA(episodes, alpha, gamma, epsilon) {
    const Q = initQ();
    const rewards = [];
    let totalCliffFalls = 0;

    for (let ep = 0; ep < episodes; ep++) {
        let r = START[0], c = START[1];
        let a = epsilonGreedy(Q, r, c, epsilon);
        let totalReward = 0;
        let steps = 0;
        const maxSteps = 10000;

        while (!isGoal(r, c) && steps < maxSteps) {
            const result = step(r, c, a);
            if (result.fellOff) totalCliffFalls++;

            // SARSA: choose next action using ε-greedy (on-policy)
            const nextA = epsilonGreedy(Q, result.nr, result.nc, epsilon);

            // SARSA update: use Q(s', a') where a' is actually chosen
            Q[r][c][a] += alpha * (result.reward + gamma * Q[result.nr][result.nc][nextA] - Q[r][c][a]);

            totalReward += result.reward;
            r = result.nr;
            c = result.nc;
            a = nextA;
            steps++;
        }
        rewards.push(totalReward);
    }

    return { Q, rewards, cliffFalls: totalCliffFalls };
}

// ── Extract greedy path from Q-table ──
function extractPath(Q) {
    const path = [];
    let r = START[0], c = START[1];
    const visited = new Set();
    const maxSteps = 100;
    let steps = 0;

    while (!isGoal(r, c) && steps < maxSteps) {
        const key = `${r},${c}`;
        if (visited.has(key)) break; // loop detection
        visited.add(key);
        path.push([r, c]);

        const a = greedyAction(Q, r, c);
        let nr = r + ACTIONS[a][0];
        let nc = c + ACTIONS[a][1];
        if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) {
            break; // stuck
        }
        if (isCliff(nr, nc)) {
            break; // policy goes through cliff
        }
        r = nr;
        c = nc;
        steps++;
    }
    if (isGoal(r, c)) path.push([r, c]);
    return path;
}

// ── Compute moving average ──
function movingAverage(data, window) {
    const result = [];
    for (let i = 0; i < data.length; i++) {
        const start = Math.max(0, i - window + 1);
        let sum = 0;
        for (let j = start; j <= i; j++) sum += data[j];
        result.push(sum / (i - start + 1));
    }
    return result;
}

// ── Compute standard deviation ──
function stdDev(arr) {
    if (arr.length === 0) return 0;
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const variance = arr.reduce((acc, val) => acc + (val - mean) ** 2, 0) / arr.length;
    return Math.sqrt(variance);
}

// ── Find convergence episode (first episode where moving avg ≥ threshold) ──
function findConvergence(rewards, threshold, window) {
    const ma = movingAverage(rewards, window);
    for (let i = 0; i < ma.length; i++) {
        if (ma[i] >= threshold) return i;
    }
    return -1;
}

// =====================================================
// UI & Rendering
// =====================================================

let chartReward = null;
let chartSmooth = null;
let trainingResults = null;

// ── Bind slider value displays ──
document.querySelectorAll('input[type="range"]').forEach(input => {
    const display = document.getElementById(input.id + '-val');
    input.addEventListener('input', () => {
        display.textContent = input.value;
    });
});

// ── Render grid with policy arrows ──
function renderGrid(containerId, Q, pathCells, pathClass) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    const pathSet = new Set(pathCells.map(p => `${p[0]},${p[1]}`));

    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.dataset.row = r;
            cell.dataset.col = c;

            if (isStart(r, c)) {
                cell.classList.add('start');
                cell.textContent = 'S';
            } else if (isGoal(r, c)) {
                cell.classList.add('goal');
                cell.textContent = 'G';
            } else if (isCliff(r, c)) {
                cell.classList.add('cliff');
                cell.textContent = '☠';
            } else {
                // Show policy arrow
                if (Q) {
                    const a = greedyAction(Q, r, c);
                    const arrow = document.createElement('span');
                    arrow.className = 'arrow';
                    arrow.textContent = ACTION_ARROWS[a];
                    cell.appendChild(arrow);
                }
                if (pathSet.has(`${r},${c}`)) {
                    cell.classList.add(pathClass);
                }
            }

            container.appendChild(cell);
        }
    }
}

// ── Initialize empty grids ──
function initGrids() {
    renderGrid('grid-qlearning', null, [], 'path-q');
    renderGrid('grid-sarsa', null, [], 'path-s');
}

// ── Render Charts ──
function renderCharts(qRewards, sRewards) {
    const labels = Array.from({ length: qRewards.length }, (_, i) => i + 1);
    const qSmooth = movingAverage(qRewards, 20);
    const sSmooth = movingAverage(sRewards, 20);

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: true,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                labels: {
                    color: '#94a3b8',
                    font: { family: 'Inter', size: 12 },
                    usePointStyle: true,
                    pointStyle: 'circle',
                }
            },
            tooltip: {
                backgroundColor: 'rgba(17, 24, 39, 0.9)',
                titleColor: '#f1f5f9',
                bodyColor: '#94a3b8',
                borderColor: 'rgba(99, 102, 241, 0.3)',
                borderWidth: 1,
                cornerRadius: 8,
                padding: 12,
            }
        },
        scales: {
            x: {
                title: { display: true, text: '回合 (Episode)', color: '#64748b', font: { family: 'Inter' } },
                ticks: { color: '#64748b', maxTicksLimit: 10 },
                grid: { color: 'rgba(255,255,255,0.04)' }
            },
            y: {
                title: { display: true, text: '累積獎勵', color: '#64748b', font: { family: 'Inter' } },
                ticks: { color: '#64748b' },
                grid: { color: 'rgba(255,255,255,0.04)' }
            }
        }
    };

    // Raw reward chart
    if (chartReward) chartReward.destroy();
    chartReward = new Chart(document.getElementById('reward-chart'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Q-learning',
                    data: qRewards,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1,
                },
                {
                    label: 'SARSA',
                    data: sRewards,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1,
                }
            ]
        },
        options: chartOptions
    });

    // Smoothed reward chart
    if (chartSmooth) chartSmooth.destroy();
    chartSmooth = new Chart(document.getElementById('smooth-reward-chart'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Q-learning (平滑)',
                    data: qSmooth,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.15)',
                    borderWidth: 2.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.3,
                },
                {
                    label: 'SARSA (平滑)',
                    data: sSmooth,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.15)',
                    borderWidth: 2.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.3,
                }
            ]
        },
        options: chartOptions
    });
}

// ── Update statistics display ──
function updateStats(qResult, sResult, qPath, sPath) {
    const lastN = 100;
    const qLast = qResult.rewards.slice(-lastN);
    const sLast = sResult.rewards.slice(-lastN);

    const qAvg = (qLast.reduce((a, b) => a + b, 0) / qLast.length).toFixed(1);
    const sAvg = (sLast.reduce((a, b) => a + b, 0) / sLast.length).toFixed(1);

    const qBest = Math.max(...qResult.rewards);
    const sBest = Math.max(...sResult.rewards);

    const qConv = findConvergence(qResult.rewards, -20, 20);
    const sConv = findConvergence(sResult.rewards, -20, 20);

    const qStd = stdDev(qLast).toFixed(1);
    const sStd = stdDev(sLast).toFixed(1);

    document.getElementById('stat-q-avg').textContent = qAvg;
    document.getElementById('stat-q-best').textContent = qBest;
    document.getElementById('stat-q-conv').textContent = qConv >= 0 ? qConv : '未收斂';
    document.getElementById('stat-q-std').textContent = qStd;
    document.getElementById('stat-q-path').textContent = qPath.length > 0 ? qPath.length : '—';
    document.getElementById('stat-q-cliff').textContent = qResult.cliffFalls;

    document.getElementById('stat-s-avg').textContent = sAvg;
    document.getElementById('stat-s-best').textContent = sBest;
    document.getElementById('stat-s-conv').textContent = sConv >= 0 ? sConv : '未收斂';
    document.getElementById('stat-s-std').textContent = sStd;
    document.getElementById('stat-s-path').textContent = sPath.length > 0 ? sPath.length : '—';
    document.getElementById('stat-s-cliff').textContent = sResult.cliffFalls;

    return { qAvg, sAvg, qBest, sBest, qConv, sConv, qStd, sStd, qPath, sPath };
}

// ── Generate Analysis Report ──
function generateAnalysis(stats, epsilon) {
    const container = document.getElementById('analysis-content');
    container.className = 'analysis-report';

    const qFaster = (stats.qConv >= 0 && stats.sConv >= 0) ? stats.qConv < stats.sConv : stats.qConv >= 0;
    const qMoreStable = parseFloat(stats.qStd) < parseFloat(stats.sStd);
    const qShorterPath = stats.qPath.length < stats.sPath.length;

    container.innerHTML = `
        <div class="analysis-block">
            <h5>📈 學習表現比較</h5>
            <ul>
                <li>Q-learning 後 100 回合平均獎勵為 <strong>${stats.qAvg}</strong>，SARSA 為 <strong>${stats.sAvg}</strong>。</li>
                <li>Q-learning 最佳單回合獎勵為 <strong>${stats.qBest}</strong>，SARSA 為 <strong>${stats.sBest}</strong>。</li>
                <li>${qFaster
                    ? `Q-learning 收斂較快（第 ${stats.qConv} 回合 vs 第 ${stats.sConv} 回合），但訓練過程中振盪較大。`
                    : `SARSA 收斂較快或兩者相當，SARSA 的訓練過程更加平穩。`
                }</li>
                <li>在 Cliff Walking 環境中，SARSA 的平均獎勵通常更高，因為它在訓練期間避免了因探索導致的懸崖掉落。</li>
            </ul>
        </div>

        <div class="analysis-block sarsa-block">
            <h5>🗺️ 策略行為分析</h5>
            <ul>
                <li>Q-learning 最終路徑長度為 <strong>${stats.qPath.length} 步</strong>，SARSA 為 <strong>${stats.sPath.length} 步</strong>。</li>
                <li>${qShorterPath
                    ? `Q-learning 學到了更短（更冒險）的路徑，傾向沿懸崖邊緣行走——這是理論最優路徑。`
                    : `兩種方法的路徑長度相近。`
                }</li>
                <li>SARSA 因為是 On-policy 方法，會考慮 ε-greedy 探索帶來的風險，因此學到的路徑通常<strong>遠離懸崖</strong>，走上方繞路。</li>
                <li>Q-learning 作為 Off-policy 方法，學習的是最優 Q 值，因此傾向<strong>沿懸崖邊走最短路徑</strong>。</li>
                <li>當 ε = ${epsilon} 時，探索機率造成 SARSA 更加「保守」，Q-learning 更加「冒險」。</li>
            </ul>
        </div>

        <div class="analysis-block">
            <h5>📉 穩定性分析</h5>
            <ul>
                <li>Q-learning 後 100 回合標準差為 <strong>${stats.qStd}</strong>，SARSA 為 <strong>${stats.sStd}</strong>。</li>
                <li>${qMoreStable
                    ? `在本次實驗中，Q-learning 的波動較小。`
                    : `SARSA 的學習過程更穩定，波動更小。`
                }</li>
                <li>Q-learning 在訓練過程中因為探索可能掉落懸崖（共 <strong>${document.getElementById('stat-q-cliff').textContent} 次</strong>），
                   導致出現大幅負獎勵尖峰，而 SARSA 掉落次數為 <strong>${document.getElementById('stat-s-cliff').textContent} 次</strong>。</li>
                <li>ε 值越大，Q-learning 的波動越劇烈，SARSA 的路徑越保守。</li>
            </ul>
        </div>

        <div class="analysis-block conclusion-block">
            <h5>🎯 結論</h5>
            <ul>
                <li><strong>收斂速度：</strong>${qFaster ? 'Q-learning 收斂較快' : 'SARSA 收斂較快或兩者相當'}，但 Q-learning 在收斂後仍有較大波動。</li>
                <li><strong>穩定性：</strong>${qMoreStable ? 'Q-learning 較穩定' : 'SARSA 較穩定'}，SARSA 通常在有懸崖/危險的環境中展現更穩定的行為。</li>
                <li><strong>適用場景：</strong>
                    <ul>
                        <li>若環境中犯錯代價低、追求理論最優策略 → 選擇 <strong>Q-learning</strong></li>
                        <li>若環境中犯錯代價高、需要安全穩定的策略 → 選擇 <strong>SARSA</strong></li>
                        <li>在真實機器人控制、自動駕駛等安全敏感場景，SARSA 類的 On-policy 方法更為適合</li>
                    </ul>
                </li>
            </ul>
        </div>
    `;
}

// ── Training entry point ──
async function startTraining() {
    const epsilon = parseFloat(document.getElementById('epsilon').value);
    const alpha = parseFloat(document.getElementById('alpha').value);
    const gamma = parseFloat(document.getElementById('gamma').value);
    const episodes = parseInt(document.getElementById('episodes').value);

    // Disable buttons
    document.getElementById('btn-train').disabled = true;
    document.getElementById('btn-animate').disabled = true;

    // Show progress
    const progressContainer = document.getElementById('progress-container');
    progressContainer.classList.remove('hidden');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    progressFill.style.width = '0%';
    progressText.textContent = '正在訓練 Q-learning...';

    // Use setTimeout to allow UI update
    await new Promise(resolve => setTimeout(resolve, 50));

    // Train Q-learning
    progressFill.style.width = '25%';
    const qResult = trainQLearning(episodes, alpha, gamma, epsilon);
    await new Promise(resolve => setTimeout(resolve, 30));

    // Train SARSA
    progressText.textContent = '正在訓練 SARSA...';
    progressFill.style.width = '55%';
    await new Promise(resolve => setTimeout(resolve, 30));
    const sResult = trainSARSA(episodes, alpha, gamma, epsilon);

    // Extract paths
    progressText.textContent = '正在分析結果...';
    progressFill.style.width = '80%';
    await new Promise(resolve => setTimeout(resolve, 30));

    const qPath = extractPath(qResult.Q);
    const sPath = extractPath(sResult.Q);

    // Render grids
    renderGrid('grid-qlearning', qResult.Q, qPath, 'path-q');
    renderGrid('grid-sarsa', sResult.Q, sPath, 'path-s');

    // Render charts
    renderCharts(qResult.rewards, sResult.rewards);

    // Update stats
    const stats = updateStats(qResult, sResult, qPath, sPath);

    // Generate analysis
    generateAnalysis(stats, epsilon);

    // Store results for animation
    trainingResults = { qResult, sResult, qPath, sPath };

    // Done
    progressFill.style.width = '100%';
    progressText.textContent = '訓練完成！';
    await new Promise(resolve => setTimeout(resolve, 500));
    progressContainer.classList.add('hidden');

    document.getElementById('btn-train').disabled = false;
    document.getElementById('btn-animate').disabled = false;

    // Smooth scroll to grids
    document.getElementById('gridworld-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Path Animation ──
async function animatePaths() {
    if (!trainingResults) return;
    document.getElementById('btn-animate').disabled = true;
    document.getElementById('btn-train').disabled = true;

    const { qResult, sResult, qPath, sPath } = trainingResults;

    // Re-render grids without path highlighting
    renderGrid('grid-qlearning', qResult.Q, [], 'path-q');
    renderGrid('grid-sarsa', sResult.Q, [], 'path-s');

    const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
    const speed = 200;

    // Animate Q-learning path
    const gridQ = document.getElementById('grid-qlearning');
    for (let i = 0; i < qPath.length; i++) {
        const [r, c] = qPath[i];
        const idx = r * COLS + c;
        const cell = gridQ.children[idx];
        if (cell) {
            // Remove previous agent
            const prev = gridQ.querySelector('.agent');
            if (prev) {
                prev.classList.remove('agent');
                // Add path class
                if (!prev.classList.contains('start') && !prev.classList.contains('goal') && !prev.classList.contains('cliff')) {
                    prev.classList.add('path-q');
                }
            }
            cell.classList.add('agent');
        }
        await delay(speed);
    }
    // Final: remove agent, show full path
    const prevQ = gridQ.querySelector('.agent');
    if (prevQ) {
        prevQ.classList.remove('agent');
        if (!prevQ.classList.contains('start') && !prevQ.classList.contains('goal') && !prevQ.classList.contains('cliff')) {
            prevQ.classList.add('path-q');
        }
    }

    await delay(300);

    // Animate SARSA path
    const gridS = document.getElementById('grid-sarsa');
    for (let i = 0; i < sPath.length; i++) {
        const [r, c] = sPath[i];
        const idx = r * COLS + c;
        const cell = gridS.children[idx];
        if (cell) {
            const prev = gridS.querySelector('.agent');
            if (prev) {
                prev.classList.remove('agent');
                if (!prev.classList.contains('start') && !prev.classList.contains('goal') && !prev.classList.contains('cliff')) {
                    prev.classList.add('path-s');
                }
            }
            cell.classList.add('agent');
        }
        await delay(speed);
    }
    const prevS = gridS.querySelector('.agent');
    if (prevS) {
        prevS.classList.remove('agent');
        if (!prevS.classList.contains('start') && !prevS.classList.contains('goal') && !prevS.classList.contains('cliff')) {
            prevS.classList.add('path-s');
        }
    }

    document.getElementById('btn-animate').disabled = false;
    document.getElementById('btn-train').disabled = false;
}

// ── Reset ──
function resetAll() {
    trainingResults = null;
    initGrids();

    if (chartReward) { chartReward.destroy(); chartReward = null; }
    if (chartSmooth) { chartSmooth.destroy(); chartSmooth = null; }

    // Clear stats
    ['stat-q-avg', 'stat-q-best', 'stat-q-conv', 'stat-q-std', 'stat-q-path', 'stat-q-cliff',
     'stat-s-avg', 'stat-s-best', 'stat-s-conv', 'stat-s-std', 'stat-s-path', 'stat-s-cliff'
    ].forEach(id => document.getElementById(id).textContent = '—');

    // Reset analysis
    const analysis = document.getElementById('analysis-content');
    analysis.className = 'analysis-placeholder';
    analysis.innerHTML = '<p>請先完成訓練以生成分析報告 📊</p>';

    document.getElementById('btn-animate').disabled = true;
    document.getElementById('progress-container').classList.add('hidden');
}

// ── Initialize on load ──
window.addEventListener('DOMContentLoaded', () => {
    initGrids();
});
