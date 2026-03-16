const canvas = document.getElementById('grid');
const ctx = canvas.getContext('2d');

const ui = {
  startBtn: document.getElementById('startBtn'),
  stepBtn: document.getElementById('stepBtn'),
  resetBtn: document.getElementById('resetBtn'),
  iterCount: document.getElementById('iterCount'),
  episodeCount: document.getElementById('episodeCount'),
  lastReward: document.getElementById('lastReward'),
  gridSize: document.getElementById('gridSize'),
  speed: document.getElementById('speed'),
  alpha: document.getElementById('alpha'),
  gamma: document.getElementById('gamma'),
  epsilon: document.getElementById('epsilon'),
  decay: document.getElementById('decay'),
  stepPenalty: document.getElementById('stepPenalty'),
  dynamicEnv: document.getElementById('dynamicEnv'),
  tools: Array.from(document.querySelectorAll('.tool')),
  modeBtn: document.getElementById('modeBtn'),
  stepTitle: document.getElementById('stepTitle'),
  stepDesc: document.getElementById('stepDesc'),
  prevStep: document.getElementById('prevStep'),
  nextStep: document.getElementById('nextStep'),
  applyStep: document.getElementById('applyStep'),
};

const ACTIONS = [
  { dx: 0, dy: -1 }, // up
  { dx: 1, dy: 0 },  // right
  { dx: 0, dy: 1 },  // down
  { dx: -1, dy: 0 }, // left
];

const rewards = {
  meat: 10,
  lava: -10,
  rock: -2,
};

const GUIDED_STEPS = [
  {
    title: 'Step 1: Learn to Move',
    desc: 'No rewards yet. The dino explores the grid and resets after a short episode.',
    config: { foods: 0, lava: 0, rocks: 0, stepPenalty: 0, epsilon: 0.2, decay: false, dynamicEnv: false, maxSteps: 40 },
  },
  {
    title: 'Step 2: Add Food',
    desc: 'Place a single food reward. The agent starts learning a path toward it.',
    config: { foods: 1, lava: 0, rocks: 0, stepPenalty: -0.02, epsilon: 0.12, decay: true, dynamicEnv: false, maxSteps: 60 },
  },
  {
    title: 'Step 3: More Food',
    desc: 'Add two food tiles. The agent learns which reward is easiest to reach.',
    config: { foods: 2, lava: 0, rocks: 0, stepPenalty: -0.02, epsilon: 0.12, decay: true, dynamicEnv: false, maxSteps: 60 },
  },
  {
    title: 'Step 4: Add Lava',
    desc: 'Introduce a lava tile. Touching it ends the episode with a negative reward.',
    config: { foods: 1, lava: 1, rocks: 0, stepPenalty: -0.02, epsilon: 0.12, decay: true, dynamicEnv: false, maxSteps: 70 },
  },
  {
    title: 'Step 5: More Lava',
    desc: 'Add more lava. The agent should find safer routes to the food.',
    config: { foods: 1, lava: 4, rocks: 0, stepPenalty: -0.03, epsilon: 0.1, decay: true, dynamicEnv: false, maxSteps: 80 },
  },
  {
    title: 'Step 6: Add Rocks',
    desc: 'Rocks are small penalties. The agent chooses the least costly path.',
    config: { foods: 1, lava: 4, rocks: 3, stepPenalty: -0.04, epsilon: 0.08, decay: true, dynamicEnv: false, maxSteps: 90 },
  },
];

const state = {
  gridSize: Number(ui.gridSize.value),
  speed: Number(ui.speed.value),
  alpha: Number(ui.alpha.value),
  gamma: Number(ui.gamma.value),
  epsilon: Number(ui.epsilon.value),
  stepPenalty: Number(ui.stepPenalty.value),
  decay: ui.decay.checked,
  dynamicEnv: ui.dynamicEnv.checked,
  training: false,
  iteration: 0,
  episode: 0,
  lastReward: 0,
  lastAction: 1,
  tool: 'meat',
  Q: null,
  frame: 0,
  renderEvery: 2,
  valueMap: null,
  valueUpdateEvery: 6,
  canvasSize: 0,
  cellSize: 0,
  start: { x: 0, y: 0 },
  dino: { x: 0, y: 0 },
  foods: new Set(),
  lava: new Set(),
  rocks: new Set(),
  mode: 'guided',
  stepIndex: 0,
  episodeSteps: 0,
  maxStepsPerEpisode: 50,
};

function key(x, y) {
  return `${x},${y}`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function randomInt(max) {
  return Math.floor(Math.random() * max);
}

function randomCell() {
  return { x: randomInt(state.gridSize), y: randomInt(state.gridSize) };
}

function isOccupied(x, y) {
  const k = key(x, y);
  if (state.start.x === x && state.start.y === y) return true;
  if (state.foods.has(k)) return true;
  if (state.lava.has(k)) return true;
  if (state.rocks.has(k)) return true;
  return false;
}

function placeRandomAvoid(avoidFn) {
  for (let i = 0; i < 200; i += 1) {
    const cell = randomCell();
    if (!avoidFn(cell.x, cell.y)) return cell;
  }
  return { x: 0, y: 0 };
}

function initQ() {
  const size = state.gridSize * state.gridSize * ACTIONS.length;
  state.Q = new Float32Array(size);
  state.valueMap = new Float32Array(state.gridSize * state.gridSize);
}

function qIndex(x, y, action) {
  return (y * state.gridSize + x) * ACTIONS.length + action;
}

function maxQ(x, y) {
  let max = -Infinity;
  let best = 0;
  for (let a = 0; a < ACTIONS.length; a += 1) {
    const value = state.Q[qIndex(x, y, a)];
    if (value > max) {
      max = value;
      best = a;
    }
  }
  return { max, best };
}

function syncParamsFromUI() {
  state.alpha = Number(ui.alpha.value);
  state.gamma = Number(ui.gamma.value);
  state.epsilon = Number(ui.epsilon.value);
  state.stepPenalty = Number(ui.stepPenalty.value);
  state.speed = Number(ui.speed.value);
  state.decay = ui.decay.checked;
  state.dynamicEnv = ui.dynamicEnv.checked;
}

function clearEnvironment() {
  state.lava.clear();
  state.rocks.clear();
  state.foods.clear();
}

function placeRandomCells(count, targetSet) {
  for (let i = 0; i < count; i += 1) {
    const cell = placeRandomAvoid(isOccupied);
    targetSet.add(key(cell.x, cell.y));
  }
}

function randomizeEnvironment() {
  clearEnvironment();

  state.start = { x: 0, y: state.gridSize - 1 };
  state.dino = { ...state.start };
  state.maxStepsPerEpisode = 80;

  placeRandomCells(1, state.foods);
  placeRandomCells(Math.min(4, state.gridSize - 2), state.lava);
  placeRandomCells(Math.min(3, state.gridSize - 3), state.rocks);
}

function resetTraining() {
  syncParamsFromUI();
  state.iteration = 0;
  state.episode = 0;
  state.lastReward = 0;
  state.lastAction = 1;
  state.episodeSteps = 0;
  state.dino = { ...state.start };
  initQ();
  updateValueMap();
  updateStats();
}

function resetWorld() {
  randomizeEnvironment();
  resetTraining();
}

function applyPreset(index) {
  const step = GUIDED_STEPS[index];
  if (!step) return;

  state.training = false;
  ui.startBtn.textContent = 'Start Training';

  const config = step.config;
  state.gridSize = Number(ui.gridSize.value);
  if (config.gridSize) {
    state.gridSize = config.gridSize;
    ui.gridSize.value = String(config.gridSize);
  }

  if (typeof config.stepPenalty === 'number') ui.stepPenalty.value = String(config.stepPenalty);
  if (typeof config.epsilon === 'number') ui.epsilon.value = String(config.epsilon);
  if (typeof config.decay === 'boolean') ui.decay.checked = config.decay;
  if (typeof config.dynamicEnv === 'boolean') ui.dynamicEnv.checked = config.dynamicEnv;
  state.maxStepsPerEpisode = typeof config.maxSteps === 'number' ? config.maxSteps : 50;

  resizeCanvas();
  clearEnvironment();
  state.start = { x: 0, y: state.gridSize - 1 };
  state.dino = { ...state.start };

  if (config.foods) placeRandomCells(config.foods, state.foods);
  if (config.lava) placeRandomCells(config.lava, state.lava);
  if (config.rocks) placeRandomCells(config.rocks, state.rocks);

  resetTraining();
}

function updateGuideUI() {
  if (state.mode === 'guided') {
    const step = GUIDED_STEPS[state.stepIndex];
    ui.stepTitle.textContent = step.title;
    ui.stepDesc.textContent = step.desc;
    ui.prevStep.disabled = state.stepIndex === 0;
    ui.nextStep.disabled = state.stepIndex === GUIDED_STEPS.length - 1;
    ui.applyStep.disabled = false;
  } else {
    ui.stepTitle.textContent = 'Free Play Mode';
    ui.stepDesc.textContent = 'Full control of the environment. Use the tools to design your own scenarios.';
    ui.prevStep.disabled = true;
    ui.nextStep.disabled = true;
    ui.applyStep.disabled = true;
  }
  ui.modeBtn.textContent = state.mode === 'guided' ? 'Switch to Free Play' : 'Switch to Guided';
}

function setMode(mode) {
  state.mode = mode;
  updateGuideUI();
  if (state.mode === 'guided') {
    applyPreset(state.stepIndex);
  }
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const size = Math.min(rect.width, rect.height);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  state.canvasSize = size;
  state.cellSize = size / state.gridSize;
}

function drawCell(x, y, cellSize, fillStyle) {
  ctx.fillStyle = fillStyle;
  ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize);
}

function drawGrid() {
  const canvasSize = state.canvasSize || canvas.getBoundingClientRect().width;
  const cellSize = state.cellSize || canvasSize / state.gridSize;
  ctx.clearRect(0, 0, canvasSize, canvasSize);

  for (let y = 0; y < state.gridSize; y += 1) {
    for (let x = 0; x < state.gridSize; x += 1) {
      const t = state.valueMap[y * state.gridSize + x];
      const hue = 120 * t; // red -> green
      drawCell(x, y, cellSize, `hsl(${hue}, 50%, 35%)`);
    }
  }

  for (const lavaCell of state.lava) {
    const [x, y] = lavaCell.split(',').map(Number);
    drawCell(x, y, cellSize, 'rgba(209, 73, 91, 0.9)');
  }

  for (const rockCell of state.rocks) {
    const [x, y] = rockCell.split(',').map(Number);
    drawCell(x, y, cellSize, 'rgba(108, 111, 122, 0.9)');
  }

  if (state.foods.size > 0) {
    ctx.fillStyle = '#ff6b6b';
    for (const foodCell of state.foods) {
      const [fx, fy] = foodCell.split(',').map(Number);
      ctx.beginPath();
      ctx.arc(
        fx * cellSize + cellSize / 2,
        fy * cellSize + cellSize / 2,
        cellSize * 0.25,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }
  }

  ctx.strokeStyle = '#88c0d0';
  ctx.lineWidth = 2;
  ctx.strokeRect(
    state.start.x * cellSize + cellSize * 0.1,
    state.start.y * cellSize + cellSize * 0.1,
    cellSize * 0.8,
    cellSize * 0.8
  );

  const centerX = state.dino.x * cellSize + cellSize / 2;
  const centerY = state.dino.y * cellSize + cellSize / 2;
  const angle = state.lastAction * (Math.PI / 2);
  ctx.save();
  ctx.translate(centerX, centerY);
  ctx.rotate(angle);
  ctx.fillStyle = '#f9c784';
  ctx.beginPath();
  ctx.moveTo(0, -cellSize * 0.28);
  ctx.lineTo(cellSize * 0.22, cellSize * 0.28);
  ctx.lineTo(-cellSize * 0.22, cellSize * 0.28);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function step(action) {
  const move = ACTIONS[action];
  const nextX = clamp(state.dino.x + move.dx, 0, state.gridSize - 1);
  const nextY = clamp(state.dino.y + move.dy, 0, state.gridSize - 1);

  let reward = state.stepPenalty;
  let terminal = false;

  const cellKey = key(nextX, nextY);
  if (state.lava.has(cellKey)) {
    reward = rewards.lava;
    terminal = true;
  } else if (state.foods.has(cellKey)) {
    reward = rewards.meat;
    terminal = true;
  } else if (state.rocks.has(cellKey)) {
    reward += rewards.rock;
  }

  return { nextX, nextY, reward, terminal };
}

function trainStep() {
  const explore = Math.random() < state.epsilon;
  const { best } = maxQ(state.dino.x, state.dino.y);
  const action = explore ? randomInt(ACTIONS.length) : best;
  state.lastAction = action;

  let { nextX, nextY, reward, terminal } = step(action);
  state.episodeSteps += 1;

  if (!terminal && state.episodeSteps >= state.maxStepsPerEpisode) {
    terminal = true;
    reward = 0;
  }
  const idx = qIndex(state.dino.x, state.dino.y, action);
  const nextValue = terminal ? 0 : maxQ(nextX, nextY).max;
  state.Q[idx] += state.alpha * (reward + state.gamma * nextValue - state.Q[idx]);

  state.lastReward = reward;
  state.iteration += 1;

  if (terminal) {
    state.episode += 1;
    state.episodeSteps = 0;
    if (state.decay) {
      state.epsilon = Math.max(0.02, state.epsilon * 0.995);
      ui.epsilon.value = state.epsilon.toFixed(2);
    }
    if (state.dynamicEnv) applyDynamicChanges();
    state.dino = { ...state.start };
  } else {
    state.dino = { x: nextX, y: nextY };
  }
  if (state.iteration % state.valueUpdateEvery === 0) {
    updateValueMap();
  }
}

function applyDynamicChanges() {
  if (Math.random() < 0.15 && state.lava.size < state.gridSize) {
    const cell = placeRandomAvoid(isOccupied);
    state.lava.add(key(cell.x, cell.y));
  }
  if (Math.random() < 0.2 && state.rocks.size > 0) {
    const rockArr = Array.from(state.rocks);
    const pick = rockArr[randomInt(rockArr.length)];
    state.rocks.delete(pick);
    const cell = placeRandomAvoid(isOccupied);
    state.rocks.add(key(cell.x, cell.y));
  }
  if (Math.random() < 0.25) {
    if (state.foods.size === 0) {
      placeRandomCells(1, state.foods);
    } else {
      const foodArr = Array.from(state.foods);
      const pick = foodArr[randomInt(foodArr.length)];
      state.foods.delete(pick);
      const cell = placeRandomAvoid(isOccupied);
      state.foods.add(key(cell.x, cell.y));
    }
  }
}

function updateStats() {
  ui.iterCount.textContent = String(state.iteration);
  ui.episodeCount.textContent = String(state.episode);
  ui.lastReward.textContent = state.lastReward.toFixed(2);
}

function updateValueMap() {
  let minV = Infinity;
  let maxV = -Infinity;
  for (let y = 0; y < state.gridSize; y += 1) {
    for (let x = 0; x < state.gridSize; x += 1) {
      const { max } = maxQ(x, y);
      minV = Math.min(minV, max);
      maxV = Math.max(maxV, max);
    }
  }
  const range = maxV - minV || 1;
  for (let y = 0; y < state.gridSize; y += 1) {
    for (let x = 0; x < state.gridSize; x += 1) {
      const { max } = maxQ(x, y);
      state.valueMap[y * state.gridSize + x] = clamp((max - minV) / range, 0, 1);
    }
  }
}

function animationLoop() {
  if (state.training) {
    state.renderEvery = state.speed > 35 ? 3 : 2;
    for (let i = 0; i < state.speed; i += 1) {
      trainStep();
    }
  }
  if (!state.training || state.frame % state.renderEvery === 0) {
    drawGrid();
  }
  state.frame += 1;
  updateStats();
  requestAnimationFrame(animationLoop);
}

function handleCanvasClick(event) {
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor(((event.clientX - rect.left) / rect.width) * state.gridSize);
  const y = Math.floor(((event.clientY - rect.top) / rect.height) * state.gridSize);
  const cellKey = key(x, y);

  if (state.tool === 'meat') {
    if (!state.lava.has(cellKey) && !state.rocks.has(cellKey)) {
      state.foods.add(cellKey);
    }
  } else if (state.tool === 'lava') {
    if (!state.rocks.has(cellKey) && !state.foods.has(cellKey)) {
      state.lava.add(cellKey);
    }
  } else if (state.tool === 'rock') {
    if (!state.lava.has(cellKey) && !state.foods.has(cellKey)) {
      state.rocks.add(cellKey);
    }
  } else if (state.tool === 'start') {
    state.start = { x, y };
    state.dino = { x, y };
  } else if (state.tool === 'erase') {
    state.lava.delete(cellKey);
    state.rocks.delete(cellKey);
    state.foods.delete(cellKey);
  }
}

function setTool(tool) {
  state.tool = tool;
  ui.tools.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.tool === tool);
  });
}

ui.startBtn.addEventListener('click', () => {
  state.training = !state.training;
  ui.startBtn.textContent = state.training ? 'Pause Training' : 'Start Training';
});

ui.stepBtn.addEventListener('click', () => {
  trainStep();
  updateStats();
  drawGrid();
});

ui.resetBtn.addEventListener('click', () => {
  state.training = false;
  ui.startBtn.textContent = 'Start Training';
  state.gridSize = Number(ui.gridSize.value);
  if (state.mode === 'guided') {
    resetTraining();
  } else {
    resetWorld();
  }
});

ui.gridSize.addEventListener('change', () => {
  state.gridSize = Number(ui.gridSize.value);
  resetWorld();
  resizeCanvas();
});

ui.speed.addEventListener('input', () => {
  state.speed = Number(ui.speed.value);
});

ui.alpha.addEventListener('input', () => {
  state.alpha = Number(ui.alpha.value);
});

ui.gamma.addEventListener('input', () => {
  state.gamma = Number(ui.gamma.value);
});

ui.epsilon.addEventListener('input', () => {
  state.epsilon = Number(ui.epsilon.value);
});

ui.stepPenalty.addEventListener('input', () => {
  state.stepPenalty = Number(ui.stepPenalty.value);
});

ui.decay.addEventListener('change', () => {
  state.decay = ui.decay.checked;
});

ui.dynamicEnv.addEventListener('change', () => {
  state.dynamicEnv = ui.dynamicEnv.checked;
});

ui.tools.forEach((btn) => {
  btn.addEventListener('click', () => setTool(btn.dataset.tool));
});

ui.prevStep.addEventListener('click', () => {
  if (state.mode !== 'guided') return;
  state.stepIndex = Math.max(0, state.stepIndex - 1);
  updateGuideUI();
  applyPreset(state.stepIndex);
});

ui.nextStep.addEventListener('click', () => {
  if (state.mode !== 'guided') return;
  state.stepIndex = Math.min(GUIDED_STEPS.length - 1, state.stepIndex + 1);
  updateGuideUI();
  applyPreset(state.stepIndex);
});

ui.applyStep.addEventListener('click', () => {
  if (state.mode !== 'guided') return;
  applyPreset(state.stepIndex);
});

ui.modeBtn.addEventListener('click', () => {
  const nextMode = state.mode === 'guided' ? 'free' : 'guided';
  setMode(nextMode);
});

window.addEventListener('resize', () => {
  resizeCanvas();
  drawGrid();
});

canvas.addEventListener('click', handleCanvasClick);

resizeCanvas();
initQ();
setTool('meat');
setMode('guided');
requestAnimationFrame(animationLoop);
