let chart = null;
let isPolling = false;
let pauseState = false;

// 経過時間のフォーマット関数 (秒 -> hh:mm:ss)
function formatTime(seconds) {
    const h = Math.floor(seconds / 3600).toString().padStart(2, '0');
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
    const s = Math.floor(seconds % 60).toString().padStart(2, '0');
    return `${h}:${m}:${s}`;
}

// チャートの初期化
function initChart() {
    const ctx = document.getElementById('fitness-chart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '最良適合度スコア (Best Fitness)',
                data: [],
                borderColor: '#06b6d4',
                backgroundColor: 'rgba(6, 182, 212, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#f3f4f6' }
                }
            }
        }
    });
}

// ダッシュボードUIの更新
function updateUI(status) {
    const runningBadge = document.getElementById('status-running');
    const btnStart = document.getElementById('btn-start');
    const btnPause = document.getElementById('btn-pause');
    const btnStop = document.getElementById('btn-stop');
    
    if (status.running) {
        if (status.paused) {
            runningBadge.textContent = "一時停止中";
            runningBadge.className = "status-val badge-paused";
            btnPause.textContent = "再開";
            pauseState = true;
        } else {
            runningBadge.textContent = "実行中";
            runningBadge.className = "status-val badge-active";
            btnPause.textContent = "一時停止";
            pauseState = false;
        }
        btnStart.disabled = true;
        btnPause.disabled = false;
        btnStop.disabled = false;
        
        // 設定入力をロック
        document.getElementById('algorithm').disabled = true;
        document.getElementById('generations').disabled = true;
        document.getElementById('population_size').disabled = true;
        document.getElementById('num_games').disabled = true;
        document.getElementById('timer_hours').disabled = true;
    } else {
        runningBadge.textContent = "未開始";
        runningBadge.className = "status-val badge-inactive";
        btnStart.disabled = false;
        btnPause.disabled = true;
        btnStop.disabled = true;
        btnPause.textContent = "一時停止";
        pauseState = false;
        
        // 設定入力のロック解除
        document.getElementById('algorithm').disabled = false;
        document.getElementById('generations').disabled = false;
        document.getElementById('population_size').disabled = false;
        document.getElementById('num_games').disabled = false;
        document.getElementById('timer_hours').disabled = false;
    }
    
    // 世代表示
    document.getElementById('status-generation').textContent = 
        `${status.current_generation} / ${status.total_generations}`;
        
    // 進捗バー
    const progressPercent = status.total_generations > 0 
        ? (status.current_generation / status.total_generations) * 100 
        : 0;
    document.getElementById('progress-bar').style.width = `${progressPercent}%`;
    
    // スコアとメッセージ
    document.getElementById('status-best-score').textContent = status.best_score.toFixed(4);
    document.getElementById('status-msg').textContent = status.message;
    
    // 時間表示
    document.getElementById('elapsed-time').textContent = `経過時間: ${formatTime(status.elapsed_time)}`;
    document.getElementById('timer-limit-display').textContent = `タイマー設定: ${formatTime(status.timer_limit)}`;
    
    // パラメータ重みの更新
    if (status.best_weights && status.best_weights.length === 10) {
        for (let i = 0; i < 10; i++) {
            document.getElementById(`w-${i}`).textContent = status.best_weights[i].toFixed(2);
        }
    }
    
    // チャートの更新
    if (status.scores_history && status.scores_history.length > 0) {
        const isPyTorch = status.algorithm === 'pytorch_rl';
        const labelPrefix = isPyTorch ? 'Eval' : 'Gen';
        const datasetLabel = isPyTorch ? '評価勝率 (Win Rate %)' : '最良適合度スコア (Best Fitness)';
        
        const labels = status.scores_history.map((_, i) => `${labelPrefix} ${i + 1}`);
        chart.data.labels = labels;
        chart.data.datasets[0].label = datasetLabel;
        chart.data.datasets[0].data = status.scores_history;
        chart.update();
    }
}

// バックエンドステータスをポーリング
async function pollStatus() {
    if (!isPolling) return;
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        updateUI(data);
    } catch (error) {
        console.error("ステータス取得エラー:", error);
    }
    setTimeout(pollStatus, 1000);
}

// イベントリスナー設定
document.getElementById('btn-start').addEventListener('click', async () => {
    const algorithm = document.getElementById('algorithm').value;
    const generations = parseInt(document.getElementById('generations').value);
    const population_size = parseInt(document.getElementById('population_size').value);
    const num_games = parseInt(document.getElementById('num_games').value);
    const timer_hours = parseFloat(document.getElementById('timer_hours').value);
    
    // 時間を秒数に変換
    const timer_limit = Math.floor(timer_hours * 3600);
    
    try {
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                algorithm,
                generations,
                population_size,
                num_games,
                timer_limit
            })
        });
        const result = await response.json();
        alert(result.message);
        
        // チャートのリセット
        chart.data.labels = [];
        chart.data.datasets[0].data = [];
        chart.update();
        
        isPolling = true;
    } catch (error) {
        alert("学習開始リクエストに失敗しました。");
    }
});

document.getElementById('btn-stop').addEventListener('click', async () => {
    if (!confirm("学習を停止しますか？（現在の世代までの最良パラメータは自動保存されています）")) return;
    try {
        const response = await fetch('/api/stop', { method: 'POST' });
        const result = await response.json();
        alert(result.message);
    } catch (error) {
        alert("停止リクエストに失敗しました。");
    }
});

document.getElementById('btn-pause').addEventListener('click', async () => {
    const nextPauseState = !pauseState;
    try {
        const response = await fetch('/api/pause', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pause: nextPauseState })
        });
        const result = await response.json();
        console.log(result.message);
    } catch (error) {
        alert("一時停止リクエストに失敗しました。");
    }
});

// ページロード時
window.addEventListener('load', () => {
    initChart();
    isPolling = true;
    pollStatus();
});
