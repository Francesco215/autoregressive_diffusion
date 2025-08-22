document.addEventListener('DOMContentLoaded', function() {
    // --- 1. Style Injection ---
    const injectStyles = () => {
        const styles = `
            #attention-svg {
                user-select: none; -webkit-user-select: none; display: block; overflow: visible;
                background-color: white; border: 1px solid #e5e7eb; border-radius: 0.75rem;
                width: 100%; height: auto;
            }
            .box { stroke-width: 1.5; cursor: default; transition: fill 0.2s ease-in-out, stroke 0.2s ease-in-out, opacity 0.5s ease-in-out; }
            .box.clean { stroke: #38bdf8; fill: #e0f2fe; }
            .box.noisy { stroke: #2dd4bf; fill: #ccfbf1; }
            .box.noisy:not(.faded-out) { cursor: pointer; }
            .box:not(.faded-out):hover { fill: #f472b6; }
            .box.related-highlight { fill: #f472b6; }
            .connection { stroke-width: 2; transition: opacity 0.5s ease-in-out, stroke 0.2s ease-in-out, stroke-width 0.2s ease-in-out; opacity: 0.8; }
            .connection.dimmed { opacity: 0.05; }
            .connection.highlighted { opacity: 1; stroke: #ec4899; stroke-width: 3.5; }
            .math-host-span { font-size: 16px; text-align: center; pointer-events: none; display: flex; justify-content: center; align-items: center; width: 100%; height: 100%; color: #374151; }
            foreignObject { transition: opacity 0.5s ease-in-out; }
            .mask-cell { stroke: #e5e7eb; stroke-width: 0.5; transition: fill 0.2s ease, opacity 0.8s ease, stroke 0.2s ease, stroke-width 0.2s ease; }
            .mask-cell.on { fill: var(--cell-color, #38bdf8); }
            .mask-cell.off { fill: transparent; }
            .mask-cell.highlighted { stroke: #ec4899; stroke-width: 2; }
            .mask-cell.highlighted.on { fill: #f472b6; }
            .mask-tick .math-host-span { font-size: 14px; }
            .click-capture-bg { cursor: pointer; }
        `;
        const styleSheet = document.createElement("style");
        styleSheet.innerText = styles;
        document.head.appendChild(styleSheet);
    };
    injectStyles();

    // --- 2. D3 Setup & Constants ---
    const svg = d3.select('#attention-svg');
    if (svg.empty()) { return; }
    svg.html('');

    const N_FRAMES_DEFAULT = 5;
    const BOX_SIZE = 60, BOX_SPACING = 35, ROW_SPACING = 300, ANIM_PADDING = 80;
    const LABEL_FO_WIDTH = 80, LABEL_FO_HEIGHT = 40, LABEL_OFFSET_Y_TOP = LABEL_FO_HEIGHT + 15, LABEL_OFFSET_Y_BOTTOM = 15;
    const MASK_PADDING = 80, MASK_LABEL_OFFSET = 60, MASK_SVG_SIZE = 800, MASK_GRID_SIZE = MASK_SVG_SIZE - MASK_PADDING * 2 - MASK_LABEL_OFFSET;
    const FADE_DURATION = 600, MOVE_DURATION = 800, STAGGER_DELAY = 100;

    // --- 3. Layout for Combined View ---
    const animWidth = ANIM_PADDING * 2 + (N_FRAMES_DEFAULT * 2) * BOX_SIZE + (N_FRAMES_DEFAULT * 2 - 1) * BOX_SPACING;
    const animHeight = ANIM_PADDING * 2 + ROW_SPACING + BOX_SIZE * 2 + LABEL_OFFSET_Y_TOP + LABEL_OFFSET_Y_BOTTOM;
    const GAP = 60;
    const totalWidth = animWidth + GAP + MASK_SVG_SIZE;
    const totalHeight = MASK_SVG_SIZE;

    svg.attr('viewBox', `0 0 ${totalWidth} ${totalHeight}`);
    svg.append('text')
        .attr('id', 'main-title')
        .attr('x', totalWidth / 2)
        .attr('y', 35)
        .attr('text-anchor', 'middle')
        .style('font-size', '28px')
        .style('font-weight', 'bold')
        .style('fill', '#1f2937');

    const animGroup = svg.append('g').attr('id', 'animation-group').attr('transform', `translate(0, ${(totalHeight - animHeight) / 2})`);
    animGroup.append('text')
        .attr('x', animWidth / 2)
        .attr('y', 70)
        .attr('text-anchor', 'middle')
        .style('font-size', '24px')
        .style('font-weight', 'bold')
        .style('fill', '#374151')
        .text('DART connectivity');

    const maskGroup = svg.append('g').attr('id', 'mask-group').attr('transform', `translate(${animWidth + GAP}, 0)`);
    maskGroup.append('text')
        .attr('x', MASK_SVG_SIZE / 2)
        .attr('y', 70)
        .attr('text-anchor', 'middle')
        .style('font-size', '24px')
        .style('font-weight', 'bold')
        .style('fill', '#374151')
        .text('DART masking');

    // --- State & SVG Groups ---
    let isAnimating = false, isAnimationComplete = false, lastAnimatedIndex = null, currentNFrames = -1, boxesData = [];
    const clickCaptureRect = animGroup.append('rect').attr('class', 'click-capture-bg').attr('width', animWidth).attr('height', animHeight).attr('fill', 'transparent');
    const linesGroup = animGroup.append('g').attr('id', 'lines-group');
    const boxesGroup = animGroup.append('g').attr('id', 'boxes-group');
    const animLabelsGroup = animGroup.append('g').attr('id', 'labels-group');
    const maskGridGroup = maskGroup.append('g').attr('id', 'mask-grid-group');
    const maskLabelsGroup = maskGroup.append('g').attr('id', 'mask-labels-group');

    // --- Logic ---
    const are_connected_training = (q_idx, k_idx, n) => {
        const q_is_clean = q_idx < n, k_is_clean = k_idx < n;
        const q_eff = q_idx % n, k_eff = k_idx % n;
        if (q_is_clean && k_is_clean) return q_eff >= k_eff;
        if (!q_is_clean && k_is_clean) return q_eff > k_eff;
        if (!q_is_clean && !k_is_clean) return q_eff === k_eff;
        return false;
    };
    const are_connected_inference_anim = (q_idx, k_idx, n, target_eff) => {
        const q_is_clean = q_idx < n, k_is_clean = k_idx < n, q_eff = q_idx % n, k_eff = k_idx % n;
        if (!q_is_clean && q_eff === target_eff) {
            if (k_is_clean && target_eff > k_eff) return true;
            if (!k_is_clean && k_eff === target_eff) return true;
        }
        return false;
    };
    const getColor = d3.scaleSequential(d3.interpolateCool).domain([0, N_FRAMES_DEFAULT]);

    // --- Drawing Functions ---
    function updateMainTitle() {
        const titleText = isAnimationComplete ? "DART during inference" : "DART during training";
        svg.select('#main-title').text(titleText);
    }

    function initializeVisualizations() {
        currentNFrames = N_FRAMES_DEFAULT;
        isAnimating = false; isAnimationComplete = false; lastAnimatedIndex = null;
        drawAttentionAnimation(currentNFrames);
        drawMaskVisualization('training', { n_frames: currentNFrames });
        updateMainTitle();
        renderKatex();
    }

    function drawAttentionAnimation(n_frames) {
        const total_frames = n_frames * 2;
        const topRowY = ANIM_PADDING + LABEL_OFFSET_Y_TOP;
        const bottomRowY = topRowY + BOX_SIZE + ROW_SPACING;
        boxesData = [];
        const noisyQueryIndices = [];
        for (let i = 0; i < total_frames; i++) {
            const isClean = i < n_frames;
            if (!isClean) noisyQueryIndices.push(i);
            const x_pos = ANIM_PADDING + i * (BOX_SIZE + BOX_SPACING);
            boxesData.push({ id: `q-${i}`, type: 'q', index: i, eff_idx: i % n_frames, isClean, x: x_pos, y: topRowY });
            boxesData.push({ id: `k-${i}`, type: 'k', index: i, eff_idx: i % n_frames, isClean, x: x_pos, y: bottomRowY });
        }
        const linesData = [];
        for (let q = 0; q < total_frames; q++) for (let k = 0; k < total_frames; k++) {
            if (are_connected_training(q, k, n_frames)) linesData.push({ id: `line-${q}-${k}`, from_q: q, to_k: k });
        }
        boxesGroup.selectAll('rect.box').data(boxesData, d => d.id).join('rect')
            .attr('class', d => `box ${d.isClean ? 'clean' : 'noisy'}`).attr('id', d => d.id)
            .attr('x', d => d.x).attr('y', d => d.y).attr('width', BOX_SIZE).attr('height', BOX_SIZE)
            .attr('rx', 8).attr('ry', 8).style('opacity', 1).attr('transform', null)
            .on('click', handleBoxClick).on('mouseover', handleMouseOver).on('mouseout', handleMouseOut);
        animLabelsGroup.selectAll('foreignObject').data(boxesData, d => d.id).join('foreignObject')
            .attr('id', d => `label-${d.id}`).attr('width', LABEL_FO_WIDTH).attr('height', LABEL_FO_HEIGHT)
            .html(d => `<span class="math-host-span">$${d.isClean ? "" : "\\tilde"}{${d.type.toUpperCase()}}_{${d.eff_idx + 1}}$</span>`)
            .attr('x', d => d.x + BOX_SIZE / 2 - LABEL_FO_WIDTH / 2)
            .attr('y', d => d.type === 'q' ? d.y - LABEL_OFFSET_Y_TOP : d.y + BOX_SIZE + LABEL_OFFSET_Y_BOTTOM)
            .style('opacity', 1).attr('transform', null);
        linesGroup.selectAll('line.connection').data(linesData, d => d.id).join('line')
            .attr('class', 'connection').attr('data-from-q', d => d.from_q).attr('data-to-k', d => d.to_k)
            .attr('x1', d => boxesData.find(b => b.id === `q-${d.from_q}`).x + BOX_SIZE / 2)
            .attr('y1', d => boxesData.find(b => b.id === `q-${d.from_q}`).y + BOX_SIZE)
            .attr('x2', d => boxesData.find(b => b.id === `k-${d.to_k}`).x + BOX_SIZE / 2)
            .attr('y2', d => boxesData.find(b => b.id === `k-${d.to_k}`).y).style('opacity', 0.8)
            .attr('stroke', d => (d.from_q >= n_frames && d.to_k < n_frames) ? getColor(noisyQueryIndices.indexOf(d.from_q)) : '#6b7280');
    }

    function drawMaskVisualization(mode, { n_frames }) {
        const size = n_frames * 2;
        const cellSize = MASK_GRID_SIZE / size;
        const maskData = [], labelData = [];
        const noisyQueryIndices = Array.from({ length: n_frames }, (_, i) => i + n_frames);
        for (let q = 0; q < size; q++) {
            for (let k = 0; k < size; k++) {
                let isOn = are_connected_training(q, k, n_frames);
                let cellColor = '#6b7280';
                if (isOn && q >= n_frames && k < n_frames) cellColor = getColor(noisyQueryIndices.indexOf(q));
                maskData.push({ q, k, isOn, id: `cell-${q}-${k}`, color: cellColor });
            }
            const latexPrefix = (q >= n_frames) ? "\\tilde" : "";
            labelData.push({ type: 'q', index: q, label: `$${latexPrefix}{Q}_{${q % n_frames + 1}}$` });
            labelData.push({ type: 'k', index: q, label: `$${latexPrefix}{K}_{${q % n_frames + 1}}$` });
        }
        maskGridGroup.selectAll('rect.mask-cell').data(maskData, d => d.id).join(
            e => e.append('rect').style('opacity', 0), u => u,
            e => e.transition().duration(FADE_DURATION).style('opacity', 0).remove()
        ).attr('class', d => `mask-cell ${d.isOn ? 'on' : 'off'}`).style('--cell-color', d => d.color)
        .transition().duration(FADE_DURATION)
        .attr('x', d => MASK_PADDING + MASK_LABEL_OFFSET + d.k * cellSize).attr('y', d => MASK_PADDING + d.q * cellSize)
        .attr('width', cellSize).attr('height', cellSize).style('opacity', 1);
        maskLabelsGroup.selectAll('.mask-tick').data(labelData, d => `${d.type}-${d.index}`).join(
            e => e.append('foreignObject').attr('class', 'mask-tick').style('opacity', 0),
            u => u,
            e => e.transition().duration(FADE_DURATION).style('opacity', 0).remove()
        )
        .attr('width', MASK_LABEL_OFFSET)
        .html(d => `<span class="math-host-span">${d.label}</span>`)
        .attr('height', cellSize)
        .transition().duration(FADE_DURATION)
        .attr('x', d => d.type === 'k' ? MASK_PADDING + MASK_LABEL_OFFSET + d.index * cellSize + cellSize / 2 - MASK_LABEL_OFFSET / 2 : MASK_PADDING)
        .attr('y', d => d.type === 'q' ? MASK_PADDING + d.index * cellSize : MASK_SVG_SIZE - MASK_PADDING - MASK_LABEL_OFFSET / 1.5)
        .style('opacity', 1);
        renderKatex();
    }

    // --- Animation Functions (EXACT PORT) ---
    function startTransitionAnimation(clickedNoisyIndex) {
        if (isAnimating) return;
        clearAllHighlights();
        isAnimating = true; isAnimationComplete = false; lastAnimatedIndex = clickedNoisyIndex;
        const n_frames = currentNFrames;
        const target_eff_idx = clickedNoisyIndex % n_frames;
        const translateX = boxesData.find(d => d.id === `q-${target_eff_idx}`).x - boxesData.find(d => d.id === `q-${clickedNoisyIndex}`).x;
        linesGroup.selectAll('line.connection').transition().duration(FADE_DURATION).style('opacity', d => {
            let keep = are_connected_inference_anim(d.from_q, d.to_k, n_frames, target_eff_idx);
            if (!keep && d.from_q < n_frames && d.to_k < n_frames && d.from_q < target_eff_idx && d.to_k < target_eff_idx) {
                if (are_connected_training(d.from_q, d.to_k, n_frames)) keep = true;
            }
            return keep ? 0.8 : 0;
        });
        d3.selectAll('#boxes-group rect, #labels-group foreignObject').transition().duration(FADE_DURATION).delay(STAGGER_DELAY)
            .style('opacity', function(d) {
                const keep = d.index === clickedNoisyIndex || (d.isClean && d.eff_idx < target_eff_idx);
                d3.select(this).classed('faded-out', !keep);
                return keep ? 1 : 0;
            });
        d3.selectAll(`#q-${clickedNoisyIndex}, #k-${clickedNoisyIndex}, #label-q-${clickedNoisyIndex}, #label-k-${clickedNoisyIndex}`)
            .transition().duration(MOVE_DURATION).delay(FADE_DURATION + STAGGER_DELAY)
            .attr('transform', `translate(${translateX}, 0)`).on('end', () => { isAnimating = false; isAnimationComplete = true; updateMainTitle(); });
        linesGroup.selectAll('line.connection').filter(d => d.from_q === clickedNoisyIndex || d.to_k === clickedNoisyIndex)
            .transition().duration(MOVE_DURATION).delay(FADE_DURATION + STAGGER_DELAY)
            .attrTween('x1', function(d) { return d.from_q !== clickedNoisyIndex ? null : d3.interpolate(parseFloat(d3.select(this).attr('x1')), parseFloat(d3.select(this).attr('x1')) + translateX); })
            .attrTween('x2', function(d) { return d.to_k !== clickedNoisyIndex ? null : d3.interpolate(parseFloat(d3.select(this).attr('x2')), parseFloat(d3.select(this).attr('x2')) + translateX); });
        const indicesToKeep = new Set();
        boxesData.forEach(d => { if (d.index === clickedNoisyIndex || (d.isClean && d.eff_idx < target_eff_idx)) indicesToKeep.add(d.index); });
        maskGridGroup.selectAll('rect.mask-cell').transition().duration(FADE_DURATION).style('opacity', d => (indicesToKeep.has(d.q) && indicesToKeep.has(d.k)) ? 1 : 0);
        maskLabelsGroup.selectAll('.mask-tick').transition().duration(FADE_DURATION).style('opacity', d => indicesToKeep.has(d.index) ? 1 : 0)
            .on('end', (d, i, nodes) => {
                if (i !== nodes.length - 1) return;
                const sorted = Array.from(indicesToKeep).sort((a, b) => a - b);
                const newSize = sorted.length, newCellSize = MASK_GRID_SIZE / newSize;
                const indexMap = new Map(sorted.map((oldIndex, newIndex) => [oldIndex, newIndex]));
                maskGridGroup.selectAll('rect.mask-cell').filter(d => indicesToKeep.has(d.q) && indicesToKeep.has(d.k)).transition().duration(MOVE_DURATION)
                    .attr('x', d => MASK_PADDING + MASK_LABEL_OFFSET + indexMap.get(d.k) * newCellSize)
                    .attr('y', d => MASK_PADDING + indexMap.get(d.q) * newCellSize)
                    .attr('width', newCellSize).attr('height', newCellSize);
                maskLabelsGroup.selectAll('.mask-tick').filter(d => indicesToKeep.has(d.index)).transition().duration(MOVE_DURATION)
                    .attr('height', newCellSize)
                    .attr('x', d => d.type === 'k' ? MASK_PADDING + MASK_LABEL_OFFSET + indexMap.get(d.index) * newCellSize + newCellSize / 2 - MASK_LABEL_OFFSET / 2 : MASK_PADDING)
                    .attr('y', d => d.type === 'q' ? MASK_PADDING + indexMap.get(d.index) * newCellSize : MASK_SVG_SIZE - MASK_PADDING - MASK_LABEL_OFFSET / 1.5);
            });
    }

    function reverseTransitionAnimation() {
        if (isAnimating || lastAnimatedIndex === null) return;
        isAnimating = true; isAnimationComplete = false;
        updateMainTitle();
        const currentLastAnimatedIndex = lastAnimatedIndex;
        d3.selectAll(`#q-${currentLastAnimatedIndex}, #k-${currentLastAnimatedIndex}, #label-q-${currentLastAnimatedIndex}, #label-k-${currentLastAnimatedIndex}`)
            .transition().duration(MOVE_DURATION).attr('transform', `translate(${BOX_SIZE/2}, 0)`).on('end', function() { d3.select(this).attr('transform', null); });
        linesGroup.selectAll('line.connection').filter(d => d.from_q === currentLastAnimatedIndex || d.to_k === currentLastAnimatedIndex)
            .transition().duration(MOVE_DURATION)
            .attrTween('x1', function(d) {
                if (d.from_q !== currentLastAnimatedIndex) return null;
                const startX = parseFloat(d3.select(this).attr('x1'));
                const endX = boxesData.find(b => b.id === `q-${d.from_q}`).x + BOX_SIZE;
                return d3.interpolate(startX, endX);
            })
            .attrTween('x2', function(d) {
                if (d.to_k !== currentLastAnimatedIndex) return null;
                const startX = parseFloat(d3.select(this).attr('x2'));
                const endX = boxesData.find(b => b.id === `k-${d.to_k}`).x + BOX_SIZE;
                return d3.interpolate(startX, endX);
            });
        d3.selectAll('#boxes-group rect, #labels-group foreignObject, #lines-group line')
            .transition().duration(FADE_DURATION).delay(MOVE_DURATION - 200)
            .style('opacity', d => (d && d.from_q !== undefined) ? 0.8 : 1)
            .on('end', (d, i, nodes) => {
                if (i === nodes.length - 1) {
                    isAnimating = false; lastAnimatedIndex = null;
                    clearAllHighlights();
                }
            });
        drawMaskVisualization('training', { n_frames: currentNFrames });
    }
    
    // --- Event Handlers ---
    function handleBoxClick(event, d) {
        if (isAnimating || isAnimationComplete) return;
        if (d.type === 'q' && !d.isClean) startTransitionAnimation(d.index);
    }
    function handleMouseOver(event, d) {
        if (isAnimating || isAnimationComplete) return;
        linesGroup.selectAll('line.connection').classed('dimmed', true);
        linesGroup.selectAll(d.type === 'q' ? `[data-from-q="${d.index}"]` : `[data-to-k="${d.index}"]`).classed('dimmed', false).classed('highlighted', true).raise();
        boxesGroup.selectAll('.box').classed('related-highlight', r_d => {
            if (d.type === 'q') return r_d.type === 'k' && are_connected_training(d.index, r_d.index, currentNFrames);
            return r_d.type === 'q' && are_connected_training(r_d.index, d.index, currentNFrames);
        });
        maskGridGroup.selectAll('rect.mask-cell').filter(c_d => (d.type === 'q' ? c_d.q : c_d.k) === d.index).classed('highlighted', true);
    }
    function handleMouseOut() { if (!isAnimating && !isAnimationComplete) clearAllHighlights(); }
    function clearAllHighlights() {
        d3.selectAll('.dimmed, .highlighted, .related-highlight').classed('dimmed highlighted related-highlight', false);
    }
    function handleSvgResetClick(event) {
        if (isAnimationComplete && !isAnimating) reverseTransitionAnimation();
    }
    function renderKatex() {
        if (window.renderMathInElement) {
            renderMathInElement(svg.node(), { delimiters: [ {left: '$', right: '$', display: false} ], throwOnError: false });
        }
    }

    // --- Initial Setup & Listeners ---
    clickCaptureRect.on('click', handleSvgResetClick);
    initializeVisualizations();
});