// 対戦ビジュアライザのフロントエンドロジック

let cardsMap = {};
let attacksMap = {};
let steps = [];
let currentStepIndex = 0;
let playInterval = null;
let allLogs = []; // 全ステップのログをまとめたもの

// エネルギータイプの色と文字の対応
const ENERGY_COLORS = {
    0: { name: '無色', color: '#9ca3af' }, // COLORLESS
    1: { name: '草', color: '#22c55e' }, // GRASS
    2: { name: '炎', color: '#ef4444' }, // FIRE
    3: { name: '水', color: '#3b82f6' }, // WATER
    4: { name: '雷', color: '#eab308' }, // LIGHTNING
    5: { name: '超', color: '#a855f7' }, // PSYCHIC
    6: { name: '闘', color: '#b45309' }, // FIGHTING
    7: { name: '悪', color: '#312e81' }, // DARKNESS
    8: { name: '鋼', color: '#64748b' }, // METAL
    9: { name: '竜', color: '#ca8a04' }, // DRAGON
    10: { name: '虹', color: 'linear-gradient(to right, red, orange, yellow, green, blue, purple)' }, // RAINBOW
    11: { name: 'R', color: '#701a75' } // TEAM_ROCKET
};

// ログタイプの定義
const LOG_TYPES = {
    0: 'SHUFFLE',
    1: 'HAS_BASIC_POKEMON',
    2: 'TURN_START',
    3: 'TURN_END',
    4: 'DRAW',
    5: 'DRAW_REVERSE',
    6: 'MOVE_CARD',
    7: 'MOVE_CARD_REVERSE',
    8: 'SWITCH',
    9: 'CHANGE',
    10: 'PLAY',
    11: 'ATTACH',
    12: 'EVOLVE',
    13: 'DEVOLVE',
    14: 'MOVE_ATTACHED',
    15: 'ATTACK',
    16: 'HP_CHANGE',
    17: 'POISONED',
    18: 'BURNED',
    19: 'ASLEEP',
    20: 'PARALYZED',
    21: 'CONFUSED',
    22: 'COIN',
    23: 'RESULT'
};

// 画面ロード時の初期化
window.addEventListener('DOMContentLoaded', async () => {
    await loadBaseData();
    setupEventListeners();
});

// カードデータとワザデータの読み込み
async function loadBaseData() {
    try {
        const [cardsRes, attacksRes] = await Promise.all([
            fetch('/api/cards').then(r => r.json()),
            fetch('/api/attacks').then(r => r.json())
        ]);
        
        if (cardsRes.status === 'success') {
            cardsRes.cards.forEach(c => {
                cardsMap[c.cardId] = c;
            });
        }
        
        if (attacksRes.status === 'success') {
            attacksRes.attacks.forEach(a => {
                attacksMap[a.attackId] = a;
            });
        }
        console.log('ベースデータロード完了:', Object.keys(cardsMap).length, '枚のカード');
    } catch (e) {
        console.error('データのロードに失敗しました', e);
    }
}

// イベントリスナーの設定
function setupEventListeners() {
    document.getElementById('btn-simulate').addEventListener('click', runSimulation);
    
    // 再生コントロール
    document.getElementById('ctrl-first').addEventListener('click', () => jumpToStep(0));
    document.getElementById('ctrl-prev').addEventListener('click', () => jumpToStep(currentStepIndex - 1));
    document.getElementById('ctrl-next').addEventListener('click', () => jumpToStep(currentStepIndex + 1));
    document.getElementById('ctrl-last').addEventListener('click', () => jumpToStep(steps.length - 1));
    
    const playBtn = document.getElementById('ctrl-play');
    playBtn.addEventListener('click', togglePlay);
    
    document.getElementById('play-speed').addEventListener('change', () => {
        if (playInterval) {
            // 再生中ならタイマーを再設定
            togglePlay();
            togglePlay();
        }
    });
}

// シミュレーションの実行
async function runSimulation() {
    const p0 = document.getElementById('agent-p0').value;
    const p1 = document.getElementById('agent-p1').value;
    
    const loading = document.getElementById('loading');
    const battleArea = document.getElementById('battle-area');
    
    loading.classList.remove('hidden');
    battleArea.classList.add('hidden');
    
    if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
        document.getElementById('ctrl-play').innerText = '▶';
    }
    
    try {
        const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ agent_p0: p0, agent_p1: p1 })
        }).then(r => r.json());
        
        if (res.status === 'success') {
            steps = res.data;
            currentStepIndex = 0;
            
            // ログの統合処理
            processAllLogs();
            
            // 勝敗結果の表示
            showBattleResult();
            
            // 画面表示
            loading.classList.add('hidden');
            battleArea.classList.remove('hidden');
            
            document.getElementById('total-steps').innerText = steps.length - 1;
            renderStep(0);
        } else {
            alert('エラー: ' + res.message);
            loading.classList.add('hidden');
        }
    } catch (e) {
        alert('通信エラーが発生しました: ' + e);
        loading.classList.add('hidden');
    }
}

// 全ステップのログを結合して整理
function processAllLogs() {
    allLogs = [];
    let cumulativeLogIndex = 0;
    
    steps.forEach((step, stepIdx) => {
        if (step.logs && step.logs.length > 0) {
            step.logs.forEach(log => {
                allLogs.push({
                    stepIndex: stepIdx,
                    logIndex: cumulativeLogIndex++,
                    data: log,
                    text: formatLogText(log)
                });
            });
        }
    });
    
    // ログリストのDOM描画
    const logList = document.getElementById('log-list');
    logList.innerHTML = '';
    
    if (allLogs.length === 0) {
        logList.innerHTML = '<p class="placeholder-text">ログはありません。</p>';
        return;
    }
    
    allLogs.forEach(item => {
        const div = document.createElement('div');
        div.className = `log-item player-${item.data.playerIndex !== null ? item.data.playerIndex : 'system'}`;
        div.id = `log-item-${item.logIndex}`;
        div.innerText = item.text;
        div.addEventListener('click', () => {
            jumpToStep(item.stepIndex);
        });
        logList.appendChild(div);
    });
}

// 勝敗結果の解析と表示
function showBattleResult() {
    const banner = document.getElementById('battle-result-banner');
    if (banner) {
        banner.classList.add('hidden');
        banner.className = 'battle-result-banner';
    }
    
    const p1Badge = document.querySelector('.your-field .player-badge');
    const p2Badge = document.querySelector('.opponent-field .player-badge');
    
    const oldMarks = document.querySelectorAll('.win-lose-status');
    oldMarks.forEach(el => el.remove());

    if (steps.length === 0) return;
    
    let resultLog = null;
    for (let i = steps.length - 1; i >= 0; i--) {
        const step = steps[i];
        if (step.logs) {
            resultLog = step.logs.find(log => log.type === 23);
            if (resultLog) break;
        }
    }
    
    if (resultLog) {
        const resultVal = resultLog.result;
        const reasonCode = resultLog.reason;
        
        const p1Mark = document.createElement('span');
        p1Mark.className = 'win-lose-status';
        const p2Mark = document.createElement('span');
        p2Mark.className = 'win-lose-status';

        if (resultVal === 0) { // P1(自分) の勝利
            let reason = '対戦ルールによる勝利';
            if (reasonCode === 1) reason = '自分のサイドカードをすべて取り切りました！';
            if (reasonCode === 2) reason = '相手の山札がなくなりました（山札切れによる勝利）！';
            if (reasonCode === 3) reason = '相手の場からポケモンが全滅しました！';
            if (reasonCode === 4) reason = 'カードの効果によって勝利しました！';
            
            if (banner) {
                banner.classList.add('win');
                banner.innerHTML = `<span style="font-size: 1.25rem; display: block; margin-bottom: 0.3rem;">🏆 対戦結果: あなたの勝利！</span><span style="font-size: 0.9rem; font-weight: normal; opacity: 0.9;">勝因: ${reason}</span>`;
            }
            
            p1Mark.innerText = ' [WIN]';
            p1Mark.style.color = '#137333';
            p1Mark.style.fontWeight = 'bold';
            if (p1Badge) p1Badge.appendChild(p1Mark);

            let oppLoseReason = '';
            if (reasonCode === 1) oppLoseReason = 'サイド取られ';
            if (reasonCode === 2) oppLoseReason = '山札切れ';
            if (reasonCode === 3) oppLoseReason = 'ポケモン全滅';
            p2Mark.innerText = ` [LOSE${oppLoseReason ? ' - ' + oppLoseReason : ''}]`;
            p2Mark.style.color = '#c5221f';
            if (p2Badge) p2Badge.appendChild(p2Mark);
            
        } else if (resultVal === 1) { // P2(相手) の勝利（自分の敗北）
            let reason = '対戦ルールによる敗北';
            if (reasonCode === 1) reason = '相手にサイドカードをすべて取られました。';
            if (reasonCode === 2) reason = '自分の山札がなくなりました（山札切れによる自滅敗北）。';
            if (reasonCode === 3) reason = '自分の場からポケモンが全滅しました。';
            if (reasonCode === 4) reason = 'カードの効果によって敗北しました。';
            
            if (banner) {
                banner.classList.add('lose');
                banner.innerHTML = `<span style="font-size: 1.25rem; display: block; margin-bottom: 0.3rem;">💀 対戦結果: あなたの敗北...</span><span style="font-size: 0.9rem; font-weight: normal; opacity: 0.9;">敗因: ${reason}</span>`;
            }
            
            let myLoseReason = '';
            if (reasonCode === 1) myLoseReason = 'サイド取られ';
            if (reasonCode === 2) myLoseReason = '山札切れ';
            if (reasonCode === 3) myLoseReason = 'ポケモン全滅';
            p1Mark.innerText = ` [LOSE${myLoseReason ? ' - ' + myLoseReason : ''}]`;
            p1Mark.style.color = '#c5221f';
            p1Mark.style.fontWeight = 'bold';
            if (p1Badge) p1Badge.appendChild(p1Mark);

            p2Mark.innerText = ' [WIN]';
            p2Mark.style.color = '#137333';
            if (p2Badge) p2Badge.appendChild(p2Mark);
            
        } else { // 引き分け
            let reason = '対戦ルールによる';
            if (reasonCode === 1) reason = 'お互いのサイドカードが同時になくなりました。';
            if (reasonCode === 2) reason = 'お互いの山札が同時になくなりました。';
            if (reasonCode === 3) reason = 'お互いのポケモンが同時に全滅しました。';
            
            if (banner) {
                banner.classList.add('draw');
                banner.innerText = `🤝 対戦結果: 引き分け (${reason})`;
            }
            
            p1Mark.innerText = ' [DRAW]';
            p1Mark.style.color = '#5f6368';
            if (p1Badge) p1Badge.appendChild(p1Mark);

            p2Mark.innerText = ' [DRAW]';
            p2Mark.style.color = '#5f6368';
            if (p2Badge) p2Badge.appendChild(p2Mark);
        }
        if (banner) banner.classList.remove('hidden');
    }
}

// ログデータを日本語テキストに変換
function formatLogText(log) {
    const pName = log.playerIndex === 0 ? 'P1(自分)' : log.playerIndex === 1 ? 'P2(相手)' : 'システム';
    const cardName = log.cardId ? (cardsMap[log.cardId] ? cardsMap[log.cardId].name : `カードID:${log.cardId}`) : '';
    const cardNameTarget = log.cardIdTarget ? (cardsMap[log.cardIdTarget] ? cardsMap[log.cardIdTarget].name : `カードID:${log.cardIdTarget}`) : '';
    
    switch (log.type) {
        case 0:
            return `${pName} はデッキをシャッフルしました。`;
        case 1:
            return `${pName} の初手チェック: たねポケモンは${log.hasBasicPokemon ? 'いました。' : 'いませんでした。'}`;
        case 2:
            return `--- ${pName} のターン開始 ---`;
        case 3:
            return `--- ${pName} のターン終了 ---`;
        case 4:
            return `${pName} はデッキから「${cardName}」をドローしました。`;
        case 5:
            return `${pName} はデッキからカードを1枚ドローしました。(非公開)`;
        case 6:
            return `${pName} は「${cardName}」を移動しました。`;
        case 7:
            return `${pName} はカードを伏せて移動しました。`;
        case 8:
            return `${pName} はバトル場の「${cardsMap[log.cardIdActive]?.name || 'ポケモン'}」とベンチの「${cardsMap[log.cardIdBench]?.name || 'ポケモン'}」を交代しました。`;
        case 9:
            return `${pName} はポケモンを「${cardName}」に変更しました。`;
        case 10:
            return `${pName} は手札から「${cardName}」をプレイしました。`;
        case 11:
            return `${pName} は「${cardNameTarget}」に「${cardName}」をつけました。`;
        case 12:
            return `${pName} は「${cardNameTarget}」を「${cardName}」に進化させました。`;
        case 13:
            return `${pName} は「${cardNameTarget}」を退化させました。`;
        case 14:
            return `${pName} はついていた「${cardName}」を移動させました。`;
        case 15:
            const attackName = log.attackId && attacksMap[log.attackId] ? attacksMap[log.attackId].name : `ワザID:${log.attackId}`;
            return `🔥 ${pName} の「${cardName}」はワザ「${attackName}」を使いました！`;
        case 16:
            const dmgText = log.value < 0 ? `${Math.abs(log.value)} ダメージ` : `HP ${log.value} 回復`;
            return `💥 「${cardName}」に ${dmgText} (残りHP変更)`;
        case 17:
            return `💀 「${cardName}」はどく状態になりました。`;
        case 18:
            return `🔥 「${cardName}」はやけど状態になりました。`;
        case 19:
            return `💤 「${cardName}」はねむり状態になりました。`;
        case 20:
            return `⚡ 「${cardName}」はまひ状態になりました。`;
        case 21:
            return `💫 「${cardName}」はこんらん状態になりました。`;
        case 22:
            return `🎲 コインフリップ: ${log.head ? 'おもて' : 'うら'}`;
        case 23:
            const winner = log.result === 0 ? 'P1(自分)' : log.result === 1 ? 'P2(相手)' : '引き分け';
            let reason = '対戦ルールによる';
            if (log.reason === 1) reason = 'サイドをすべて取りました';
            if (log.reason === 2) reason = '山札がなくなりました';
            if (log.reason === 3) reason = 'バトル場のポケモンがいなくなりました';
            if (log.reason === 4) reason = 'カード効果による';
            return `🏆 試合終了！勝者: ${winner} (${reason})`;
        default:
            return `[${LOG_TYPES[log.type]}] ${pName} 関連イベント`;
    }
}

// 特定のステップのレンダリング
function renderStep(index) {
    if (index < 0 || index >= steps.length) return;
    currentStepIndex = index;
    
    document.getElementById('current-step').innerText = index;
    
    // コントロールボタンの状態更新
    document.getElementById('ctrl-first').disabled = index === 0;
    document.getElementById('ctrl-prev').disabled = index === 0;
    document.getElementById('ctrl-next').disabled = index === steps.length - 1;
    document.getElementById('ctrl-last').disabled = index === steps.length - 1;
    
    const step = steps[index];
    const current = step.current;
    
    if (!current) {
        // 初期状態など盤面がない場合
        showEmptyBoard();
        highlightLogsForStep(index);
        return;
    }
    
    // 各プレイヤー情報の抽出 (0が自分、1が相手)
    const me = current.players[0];
    const opp = current.players[1];
    
    // 盤面描画
    renderPlayerField(me, 'your');
    renderPlayerField(opp, 'opp');
    
    // スタジアムの描画
    renderStadium(current.stadium);
    
    // ログのハイライト更新
    highlightLogsForStep(index);
}

// 盤面情報がない初期ステップの表示
function showEmptyBoard() {
    document.getElementById('your-hand').innerHTML = '<p class="placeholder-text">デッキ構築中...</p>';
    document.getElementById('opp-hand').innerHTML = '';
    document.getElementById('your-active').innerHTML = '';
    document.getElementById('opp-active').innerHTML = '';
    document.getElementById('your-bench').innerHTML = '';
    document.getElementById('opp-bench').innerHTML = '';
    document.getElementById('your-prizes').innerHTML = '';
    document.getElementById('opp-prizes').innerHTML = '';
    
    document.getElementById('your-deck').querySelector('.count').innerText = '60';
    document.getElementById('opp-deck').querySelector('.count').innerText = '60';
    document.getElementById('your-discard').querySelector('.count').innerText = '0';
    document.getElementById('opp-discard').querySelector('.count').innerText = '0';
}

// プレイヤーごとの盤面レンダリング
function renderPlayerField(player, prefix) {
    // 1. 手札のレンダリング
    const handContainer = document.getElementById(`${prefix}-hand`);
    handContainer.innerHTML = '';
    
    if (player.hand) {
        // 自分の手札 (表向き)
        player.hand.forEach(card => {
            const cardEl = createCardElement(card);
            handContainer.appendChild(cardEl);
        });
    } else {
        // 相手の手札 (裏向き)
        for (let i = 0; i < player.handCount; i++) {
            const cardBack = document.createElement('div');
            cardBack.className = 'card-back';
            handContainer.appendChild(cardBack);
        }
    }
    
    // 2. バトル場 (Active)
    const activeSlot = document.getElementById(`${prefix}-active`);
    activeSlot.innerHTML = '';
    if (player.active && player.active.length > 0 && player.active[0] !== null) {
        const pokeEl = createActivePokemonElement(player.active[0], player);
        activeSlot.appendChild(pokeEl);
    } else {
        activeSlot.innerHTML = '<span class="placeholder-text" style="font-size:0.6rem;margin-top:0;">バトル場</span>';
    }
    
    // 3. ベンチ
    const benchContainer = document.getElementById(`${prefix}-bench`);
    benchContainer.innerHTML = '';
    // 最大ベンチ枠分ループ (通常は5枠)
    const maxBench = player.benchMax || 5;
    for (let i = 0; i < maxBench; i++) {
        const slot = document.createElement('div');
        slot.className = 'active-slot'; // 同じ大きさの枠
        slot.style.width = '75px';
        slot.style.height = '100px';
        
        if (player.bench && player.bench[i]) {
            const pokeEl = createActivePokemonElement(player.bench[i], player, false);
            slot.appendChild(pokeEl);
        } else {
            slot.innerHTML = '<span class="placeholder-text" style="font-size:0.5rem;margin-top:0;">ベンチ</span>';
        }
        benchContainer.appendChild(slot);
    }
    
    // 4. サイド (賞品カード)
    const prizeContainer = document.getElementById(`${prefix}-prizes`);
    prizeContainer.innerHTML = '';
    
    // 賞品カード枠 (最大6枚)
    // 履歴データにあるサイドカードの数、または None で裏向き表示
    const totalPrizes = player.prize.length;
    for (let i = 0; i < 6; i++) {
        const item = document.createElement('div');
        if (i < totalPrizes) {
            item.className = 'side-card-item';
            // もしサイドカードが表向き (公開) されているならその情報を使う
            const pCard = player.prize[i];
            if (pCard) {
                item.title = cardsMap[pCard.id]?.name || '賞品カード';
            } else {
                item.title = '未獲得サイド';
            }
        } else {
            item.className = 'side-card-item empty';
        }
        prizeContainer.appendChild(item);
    }
    
    // 5. デッキとトラッシュ
    const deckCount = player.deckCount;
    document.getElementById(`${prefix}-deck`).querySelector('.count').innerText = deckCount;
    
    const discardCount = player.discard.length;
    const discardPile = document.getElementById(`${prefix}-discard`);
    discardPile.querySelector('.count').innerText = discardCount;
    if (discardCount > 0) {
        discardPile.classList.add('has-cards');
        // 一番上のトラッシュカードをツールチップに表示できるように
        const topDiscard = player.discard[player.discard.length - 1];
        discardPile.title = `一番上のトラッシュ: ${cardsMap[topDiscard.id]?.name || '不明'}`;
        
        // クリックでトラッシュ一覧を表示できるように設定
        discardPile.onclick = () => showDiscardList(player.discard);
    } else {
        discardPile.classList.remove('has-cards');
        discardPile.title = 'トラッシュは空です';
        discardPile.onclick = null;
    }
}

// カードDOM要素の生成 (手札用など)
function createCardElement(card) {
    const div = document.createElement('div');
    const meta = cardsMap[card.id] || { name: `ID:${card.id}`, cardType: 0 };
    
    // タイプに応じたクラス
    let typeClass = 'item';
    if (meta.cardType === 0) typeClass = 'pokemon';
    if (meta.cardType === 3) typeClass = 'supporter';
    if (meta.cardType === 5 || meta.cardType === 6) typeClass = 'energy';
    
    div.className = `card-item ${typeClass}`;
    div.dataset.id = card.id;
    
    // 名称
    const nameSpan = document.createElement('span');
    nameSpan.className = 'card-name';
    nameSpan.innerText = meta.name;
    div.appendChild(nameSpan);
    
    // HP (ポケモンの場合)
    if (meta.cardType === 0 && meta.hp) {
        const hpSpan = document.createElement('span');
        hpSpan.className = 'card-hp';
        hpSpan.innerText = `${meta.hp}HP`;
        div.appendChild(hpSpan);
    }
    
    // タイプバッジ
    const badge = document.createElement('span');
    badge.className = 'card-type-badge';
    badge.innerText = getTypeName(meta.cardType);
    div.appendChild(badge);
    
    // ホバー時の詳細表示
    div.addEventListener('mouseenter', () => showCardDetails(card.id));
    div.addEventListener('click', () => showCardDetails(card.id));
    
    return div;
}

// ポケモンタイプの日本語化
function getTypeName(typeId) {
    switch (typeId) {
        case 0: return 'ポケモン';
        case 1: return 'グッズ';
        case 2: return 'ポケモンのどうぐ';
        case 3: return 'サポート';
        case 4: return 'スタジアム';
        case 5: return '基本エネルギー';
        case 6: return '特殊エネルギー';
        default: return 'カード';
    }
}

// 場にあるポケモン（Active / Bench）のDOM要素生成
function createActivePokemonElement(pokemon, player, isActive = true) {
    const div = document.createElement('div');
    div.className = 'board-pokemon';
    div.style.width = isActive ? '80px' : '70px';
    div.style.height = isActive ? '110px' : '95px';
    
    const meta = cardsMap[pokemon.id] || { name: `ID:${pokemon.id}` };
    
    // ポケモン名
    const nameEl = document.createElement('div');
    nameEl.className = 'poke-name';
    nameEl.innerText = meta.name;
    nameEl.style.fontSize = isActive ? '0.75rem' : '0.65rem';
    div.appendChild(nameEl);
    
    // HPバー
    const hpBarContainer = document.createElement('div');
    hpBarContainer.className = 'hp-bar-container';
    
    const hpPercent = (pokemon.hp / pokemon.maxHp) * 100;
    const hpBar = document.createElement('div');
    hpBar.className = 'hp-bar';
    hpBar.style.width = `${Math.max(0, Math.min(100, hpPercent))}%`;
    
    if (hpPercent < 25) {
        hpBar.classList.add('danger');
    } else if (hpPercent < 50) {
        hpBar.classList.add('warning');
    }
    hpBarContainer.appendChild(hpBar);
    div.appendChild(hpBarContainer);
    
    // HP数値
    const hpText = document.createElement('div');
    hpText.className = 'hp-text';
    hpText.innerText = `${pokemon.hp}/${pokemon.maxHp}`;
    div.appendChild(hpText);
    
    // 付加されているエネルギーの描画
    if (pokemon.energies && pokemon.energies.length > 0) {
        const energyList = document.createElement('div');
        energyList.className = 'energy-list';
        
        pokemon.energies.forEach(eType => {
            const dot = document.createElement('div');
            dot.className = 'energy-dot';
            const eInfo = ENERGY_COLORS[eType] || { color: '#ccc' };
            dot.style.background = eInfo.color;
            dot.title = eInfo.name;
            energyList.appendChild(dot);
        });
        div.appendChild(energyList);
    }
    
    // 状態異常バッジ (毒・やけどなど)
    // バトル場のみ状態異常が存在する
    if (isActive) {
        const badgesContainer = document.createElement('div');
        badgesContainer.className = 'status-badges';
        
        if (player.poisoned) addStatusBadge(badgesContainer, '毒', 'poison');
        if (player.burned) addStatusBadge(badgesContainer, '炎', 'burn');
        if (player.asleep) addStatusBadge(badgesContainer, '眠', 'sleep');
        if (player.paralyzed) addStatusBadge(badgesContainer, '麻', 'paralyze');
        if (player.confused) addStatusBadge(badgesContainer, '乱', 'confuse');
        
        if (badgesContainer.children.length > 0) {
            div.appendChild(badgesContainer);
        }
    }
    
    // ホバー/クリックで詳細を表示
    div.addEventListener('mouseenter', () => showCardDetails(pokemon.id, pokemon));
    div.addEventListener('click', () => showCardDetails(pokemon.id, pokemon));
    
    return div;
}

function addStatusBadge(container, text, className) {
    const badge = document.createElement('span');
    badge.className = `badge-status ${className}`;
    badge.innerText = text;
    container.appendChild(badge);
}

// スタジアムの描画
function renderStadium(stadiumList) {
    const stadiumArea = document.getElementById('stadium-area');
    stadiumArea.innerHTML = '';
    
    if (stadiumList && stadiumList.length > 0) {
        const card = stadiumList[0];
        const meta = cardsMap[card.id] || { name: `スタジアム ID:${card.id}` };
        
        const div = document.createElement('div');
        div.className = 'card-item supporter'; // スタジアム用の代用スタイル
        div.style.width = '120px';
        div.style.height = '60px';
        div.style.display = 'flex';
        div.style.justifyContent = 'center';
        div.style.alignItems = 'center';
        div.style.border = '1px solid #10b981';
        
        const nameSpan = document.createElement('span');
        nameSpan.className = 'card-name';
        nameSpan.innerText = `🏟️ ${meta.name}`;
        nameSpan.style.textAlign = 'center';
        nameSpan.style.position = 'static';
        div.appendChild(nameSpan);
        
        div.addEventListener('mouseenter', () => showCardDetails(card.id));
        div.addEventListener('click', () => showCardDetails(card.id));
        stadiumArea.appendChild(div);
    } else {
        stadiumArea.innerHTML = '<div class="stadium-placeholder">スタジアムなし</div>';
    }
}

// 右側パネルにカード詳細を表示
function showCardDetails(cardId, pokemonState = null) {
    const content = document.getElementById('card-detail-content');
    const meta = cardsMap[cardId];
    
    if (!meta) {
        content.innerHTML = `<p class="placeholder-text">不明なカード (ID: ${cardId})</p>`;
        return;
    }
    
    let html = `
        <div class="detail-card">
            <div class="detail-header">
                <span class="detail-name">${meta.name}</span>
                <span class="detail-type">${getTypeName(meta.cardType)}</span>
            </div>
            <div class="detail-stats">
    `;
    
    if (meta.cardType === 0) { // ポケモン
        html += `<span>HP: ${meta.hp}</span>`;
        if (meta.evolvesFrom) {
            html += `<span>${meta.evolvesFrom} から進化</span>`;
        } else {
            html += `<span>たねポケモン</span>`;
        }
        html += `<span>にげる: ${meta.retreatCost}エネ</span>`;
    }
    
    html += `</div>`;
    
    // ポケモンの動的状態表示 (HPやエネルギーなど)
    if (pokemonState) {
        html += `
            <div style="background:rgba(255,255,255,0.05); padding:0.4rem; border-radius:6px; font-size:0.8rem; margin:0.4rem 0;">
                <strong>盤面状態:</strong><br>
                現在のHP: ${pokemonState.hp} / ${pokemonState.maxHp}<br>
                付加エネルギー: ${pokemonState.energies.map(e => ENERGY_COLORS[e]?.name || e).join(', ') || 'なし'}
            </div>
        `;
    }
    
    // スキル / 効果
    if (meta.skills && meta.skills.length > 0) {
        html += `<div class="detail-skills"><strong>特性・効果:</strong>`;
        meta.skills.forEach(s => {
            html += `
                <div class="skill-item">
                    <div class="skill-name">✨ ${s.name}</div>
                    <div class="skill-text">${s.text}</div>
                </div>
            `;
        });
        html += `</div>`;
    }
    
    // ワザの一覧 (ポケモンの場合)
    if (meta.attacks && meta.attacks.length > 0) {
        html += `<div class="detail-skills"><strong>ワザ:</strong>`;
        meta.attacks.forEach(attId => {
            const att = attacksMap[attId];
            if (att) {
                const reqEnergy = att.energies.map(e => ENERGY_COLORS[e]?.name || e).join(', ') || 'なし';
                html += `
                    <div class="skill-item">
                        <div class="detail-header" style="border:none;padding:0;">
                            <span class="skill-name" style="color:#f43f5e;">💥 ${att.name}</span>
                            <span style="font-size:0.8rem;font-weight:bold;">${att.damage > 0 ? att.damage : ''}</span>
                        </div>
                        <div class="skill-text" style="font-size:0.7rem;color:var(--text-secondary);margin-bottom:0.2rem;">
                            必要エネルギー: ${reqEnergy}
                        </div>
                        <div class="skill-text">${att.text}</div>
                    </div>
                `;
            }
        });
        html += `</div>`;
    }
    
    html += `</div>`;
    content.innerHTML = html;
}

// トラッシュリストをモーダル表示
function showDiscardList(discardCards) {
    const content = document.getElementById('card-detail-content');
    let html = `
        <div class="detail-card">
            <div class="detail-header">
                <span class="detail-name">トラッシュ一覧 (${discardCards.length}枚)</span>
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:0.4rem; max-height:200px; overflow-y:auto; padding:0.4rem; background:rgba(0,0,0,0.2); border-radius:6px;">
    `;
    
    discardCards.forEach(c => {
        const meta = cardsMap[c.id] || { name: `ID:${c.id}` };
        html += `
            <div class="badge-status" style="background:#475569; cursor:pointer;" onclick="showCardDetails(${c.id})">
                ${meta.name}
            </div>
        `;
    });
    
    html += `
            </div>
            <p style="font-size:0.7rem; color:var(--text-secondary); margin-top:0.4rem;">
                ※ カード名をクリックすると詳細情報が表示されます。
            </p>
        </div>
    `;
    content.innerHTML = html;
}

// ログ一覧のハイライト処理とスクロール
function highlightLogsForStep(stepIndex) {
    // すべてのハイライトを外す
    document.querySelectorAll('.log-list .log-item').forEach(el => {
        el.classList.remove('active');
    });
    
    // このステップに関連するログをハイライトする
    const activeItems = allLogs.filter(item => item.stepIndex === stepIndex);
    if (activeItems.length > 0) {
        // 最もインデックスの若いものにスクロール
        const targetLogIndex = activeItems[0].logIndex;
        const targetEl = document.getElementById(`log-item-${targetLogIndex}`);
        
        if (targetEl) {
            targetEl.classList.add('active');
            // 他の関連するログもアクティブにする
            activeItems.forEach(item => {
                const el = document.getElementById(`log-item-${item.logIndex}`);
                if (el) el.classList.add('active');
            });
            
            // スクロール追従
            targetEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
}

// 指定したステップへのジャンプ
function jumpToStep(index) {
    if (index < 0 || index >= steps.length) return;
    
    if (playInterval) {
        togglePlay(); // 再生中なら一時停止
    }
    
    renderStep(index);
}

// 自動再生のトグル
function togglePlay() {
    const playBtn = document.getElementById('ctrl-play');
    
    if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
        playBtn.innerText = '▶';
    } else {
        playBtn.innerText = '⏸';
        const speed = parseInt(document.getElementById('play-speed').value);
        
        playInterval = setInterval(() => {
            if (currentStepIndex < steps.length - 1) {
                renderStep(currentStepIndex + 1);
            } else {
                togglePlay(); // 最後まで行ったら停止
            }
        }, speed);
    }
}
