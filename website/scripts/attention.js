document.addEventListener('DOMContentLoaded', function() {
    // --- 1. Style Injection ---
    // Inject the necessary CSS for the animation into the document's head.
    // This makes the script self-contained and doesn't require manual CSS editing.
    const injectStyles = () => {
        const styles = `
            /* Basic styles for the visualization */
            #attention-svg {
                user-select: none;
                -webkit-user-select: none;
                display: block;
                overflow: visible;
                background-color: white;
                border: 1px solid #e5e7eb; /* gray-200 */
                border-radius: 0.75rem;
                width: 100%;
                height: auto;
            }
            /* Box styles for the main animation */
            .box {
                stroke-width: 1.5;
                cursor: default;
                transition: fill 0.2s ease-in-out, stroke 0.2s ease-in-out, opacity 0.5s ease-in-out;
            }
            .box.clean {
                stroke: #38bdf8; /* sky-400 */
                fill: #e0f2fe;   /* sky-50 */
            }
            .box.noisy {
                stroke: #2dd4bf; /* teal-400 */
                fill: #ccfbf1;   /* teal-50 */
            }
            .box.noisy:not(.faded-out) {
                cursor: pointer;
            }
            .box:not(.faded-out):hover {
                fill: #f472b6; /* pink-400 */
            }
            .box.related-highlight {
                fill: #f472b6; /* pink-400, same as hover */
            }
            /* Connection line styles */
            .connection {
                stroke-width: 2;
                transition: opacity 0.5s ease-in-out, stroke 0.2s ease-in-out, stroke-width 0.2s ease-in-out;
                opacity: 0.8;
            }
            .connection.dimmed {
                opacity: 0.05;
            }
            .connection.highlighted {
                opacity: 1;
                stroke: #ec4899; /* pink-500 */
                stroke-width: 3.5;
            }
            /* MathJax label container styles */
            .math-host-span {
                font-size: 16px;
                text-align: center;
                padding: 2px;
                transition: color 0.2s ease-in-out, opacity 0.5s ease-in-out;
                pointer-events: none;
                display: flex;
                justify-content: center;
                align-items: center;
                width: 100%;
                height: 100%;
                box-sizing: border-box;
                color: #374151; /* gray-700 */
            }
            foreignObject {
                transition: opacity 0.5s ease-in-out;
            }
            /* Styles for the mask visualization */
            .mask-cell {
                stroke: #e5e7eb; /* gray-200 */
                stroke-width: 0.5;
                transition: fill 0.2s ease-in-out, opacity 0.8s ease-in-out, stroke 0.2s ease-in-out, stroke-width 0.2s ease-in-out;
            }
            .mask-cell.on {
                fill: var(--cell-color, #38bdf8); /* Use CSS variable for dynamic color */
            }
            .mask-cell.off {
                fill: transparent;
            }
            .mask-cell.highlighted {
                stroke: #ec4899; /* pink-500 */
                stroke-width: 2;
            }
            .mask-cell.highlighted.on {
                fill: #f472b6; /* pink-400 on highlight */
            }
            .mask-tick .math-host-span {
                font-size: 14px;
            }
        `;
        const styleSheet = document.createElement("style");
        styleSheet.innerText = styles;
        document.head.appendChild(styleSheet);
    };
    injectStyles();

    // --- 2. D3 Setup and Constants ---
    const svg = d3.select('#attention-svg');
    if (svg.empty()) {
        console.error('Target SVG #attention-svg not found.');
        return;
    }
    svg.html(''); // Clear any existing content

    const N_FRAMES_DEFAULT = 5;

    // Animation constants
    const BOX_SIZE = 60;
    const BOX_SPACING = 35;
    const ROW_SPACING = 300;
    const ANIM_PADDING = 80;
    const LABEL_FO_WIDTH = 80;
    const LABEL_FO_HEIGHT = 40;
    const LABEL_OFFSET_Y_TOP = LABEL_FO_HEIGHT + 15;
    const LABEL_OFFSET_Y_BOTTOM = 15;

    // Mask constants
    const MASK_PADDING = 80;
    const MASK_LABEL_OFFSET = 60;
    const MASK_SVG_SIZE = 800;
    const MASK_GRID_SIZE = MASK_SVG_SIZE - MASK_PADDING * 2 - MASK_LABEL_OFFSET;

    // General transition constants
    const FADE_DURATION = 600;
    const MOVE_DURATION = 800;
    const STAGGER_DELAY = 100;
    
    // --- 3. Layout Calculation for Combined SVG ---
    const animWidth = ANIM_PADDING * 2 + (N_FRAMES_DEFAULT * 2) * BOX_SIZE + (N_FRAMES_DEFAULT * 2 - 1) * BOX_SPACING;
    const animHeight = ANIM_PADDING * 2 + ROW_SPACING + BOX_SIZE * 2 + LABEL_OFFSET_Y_TOP + LABEL_OFFSET_Y_BOTTOM;
    const GAP = 60;
    const totalWidth = animWidth + GAP + MASK_SVG_SIZE;
    const totalHeight = MASK_SVG_SIZE; // Use the taller visualization's height
    
    svg.attr('viewBox', `0 0 ${totalWidth} ${totalHeight}`);

    // Center the animation vertically
    const animGroupYOffset = (totalHeight - animHeight) / 2;
    const maskGroupXOffset = animWidth + GAP;

    const animGroup = svg.append('g').attr('id', 'animation-group').attr('transform', `translate(0, ${animGroupYOffset})`);
    const maskGroup = svg.append('g').attr('id', 'mask-visualization-group').attr('transform', `translate(${maskGroupXOffset}, 0)`);

    // --- State Management ---
    let isAnimating = false;
    let isAnimationComplete = false;
    let lastAnimatedIndex = null;
    let currentNFrames = -1;
    let boxesData = [];

    // --- Groups for SVG layers ---
    const linesGroup = animGroup.append('g').attr('id', 'lines-group');
    const boxesGroup = animGroup.append('g').attr('id', 'boxes-group');
    const animLabelsGroup = animGroup.append('g').attr('id', 'labels-group');
    const maskGridGroup = maskGroup.append('g').attr('id', 'mask-grid-group');
    const maskLabelsGroup = maskGroup.append('g').attr('id', 'mask-labels-group');

    // --- Connectivity Logic ---
    const are_connected_training = (q_idx, k_idx, n) => {
        const q_is_clean = q_idx < n;
        const k_is_clean = k_idx < n;
        const q_eff = q_idx % n;
        const k_eff = k_idx % n;

        if (q_is_clean && k_is_clean) return q_eff >= k_eff;
        if (!q_is_clean && k_is_clean) return q_eff > k_eff;
        if (!q_is_clean && !k_is_clean) return q_eff === k_eff;
        return false;
    };

    const are_connected_inference_anim = (q_idx, k_idx, n, target_eff) => {
        const q_is_clean = q_idx < n;
        const k_is_clean = k_idx < n;
        const q_eff = q_idx % n;
        const k_eff = k_idx % n;
        if (!q_is_clean && q_eff === target_eff) {
            if (k_is_clean && target_eff > k_eff) return true;
            if (!k_is_clean && k_eff === target_eff) return true;
        }
        return false;
    };

    // --- Color Palette ---
    const getColor = d3.scaleSequential(d3.interpolateCool).domain([0, N_FRAMES_DEFAULT]);

    // --- Main Drawing Function ---
    function initializeVisualizations() {
        const n_frames = N_FRAMES_DEFAULT;
        currentNFrames = n_frames;

        isAnimating = false;
        isAnimationComplete = false;
        lastAnimatedIndex = null;

        drawAttentionAnimation(n_frames);
        drawMaskVisualization('training', { n_frames });

        renderKatex();
    }

    // --- Attention Animation Drawing ---
    function drawAttentionAnimation(n_frames) {
        const total_frames = n_frames * 2;
        const topRowY = ANIM_PADDING + LABEL_OFFSET_Y_TOP;
        const bottomRowY = topRowY + BOX_SIZE + ROW_SPACING;

        boxesData = [];
        const noisyQueryIndices = [];
        for (let i = 0; i < total_frames; i++) {
            const isClean = i < n_frames;
            if (!isClean) noisyQueryIndices.push(i);
            const eff_idx = i % n_frames;
            const x_pos = ANIM_PADDING + i * (BOX_SIZE + BOX_SPACING);

            boxesData.push({ id: `q-${i}`, type: 'q', index: i, eff_idx, isClean, x: x_pos, y: topRowY });
            boxesData.push({ id: `k-${i}`, type: 'k', index: i, eff_idx, isClean, x: x_pos, y: bottomRowY });
        }

        const linesData = [];
        for (let q_idx = 0; q_idx < total_frames; q_idx++) {
            for (let k_idx = 0; k_idx < total_frames; k_idx++) {
                if (are_connected_training(q_idx, k_idx, n_frames)) {
                    linesData.push({ id: `line-${q_idx}-${k_idx}`, from_q: q_idx, to_k: k_idx });
                }
            }
        }

        boxesGroup.selectAll('rect.box')
            .data(boxesData, d => d.id)
            .join('rect')
            .attr('class', d => `box ${d.isClean ? 'clean' : 'noisy'}`)
            .attr('id', d => d.id)
            .attr('x', d => d.x).attr('y', d => d.y)
            .attr('width', BOX_SIZE).attr('height', BOX_SIZE)
            .attr('rx', 8).attr('ry', 8)
            .style('opacity', 1).attr('transform', null)
            .on('click', handleBoxClick)
            .on('mouseover', handleMouseOver)
            .on('mouseout', handleMouseOut);

        animLabelsGroup.selectAll('foreignObject')
            .data(boxesData, d => d.id)
            .join('foreignObject')
            .attr('id', d => `label-${d.id}`)
            .attr('width', LABEL_FO_WIDTH).attr('height', LABEL_FO_HEIGHT)
            .html(d => `<span class="math-host-span">$${d.isClean ? "" : "\\tilde"}{${d.type.toUpperCase()}}_{${d.eff_idx + 1}}$</span>`)
            .attr('x', d => d.x + BOX_SIZE / 2 - LABEL_FO_WIDTH / 2)
            .attr('y', d => d.type === 'q' ? d.y - LABEL_OFFSET_Y_TOP : d.y + BOX_SIZE + LABEL_OFFSET_Y_BOTTOM)
            .style('opacity', 1).attr('transform', null);

        linesGroup.selectAll('line.connection')
            .data(linesData, d => d.id)
            .join('line')
            .attr('class', 'connection')
            .attr('data-from-q', d => d.from_q).attr('data-to-k', d => d.to_k)
            .attr('x1', d => boxesData.find(b => b.id === `q-${d.from_q}`).x + BOX_SIZE / 2)
            .attr('y1', d => boxesData.find(b => b.id === `q-${d.from_q}`).y + BOX_SIZE)
            .attr('x2', d => boxesData.find(b => b.id === `k-${d.to_k}`).x + BOX_SIZE / 2)
            .attr('y2', d => boxesData.find(b => b.id === `k-${d.to_k}`).y)
            .style('opacity', 0.8)
            .attr('stroke', d => {
                const q_is_clean = d.from_q < n_frames;
                const k_is_clean = d.to_k < n_frames;
                if (!q_is_clean && k_is_clean) {
                    return getColor(noisyQueryIndices.indexOf(d.from_q));
                }
                return '#6b7280'; // gray-500
            });
    }

    // --- Mask Visualization Drawing ---
    function drawMaskVisualization(mode, { n_frames, target_eff_idx = 0 }) {
        const size = mode === 'training' ? n_frames * 2 : target_eff_idx + 1;
        const cellSize = MASK_GRID_SIZE / size;

        const maskData = [];
        const labelData = [];
        const noisyQueryIndices = mode === 'training' ? Array.from({ length: n_frames }, (_, i) => i + n_frames) : [];

        for (let q = 0; q < size; q++) {
            for (let k = 0; k < size; k++) {
                let isOn = (mode === 'training') ? are_connected_training(q, k, n_frames) : (q >= k);
                let cellColor = 'transparent';
                if (isOn) {
                    if (mode === 'training') {
                        if (q >= n_frames && k < n_frames) { // Noisy Query, Clean Key
                            cellColor = getColor(noisyQueryIndices.indexOf(q));
                        } else {
                            cellColor = '#6b7280'; // gray
                        }
                    } else { // inference mode
                        cellColor = '#6b7280';
                    }
                }
                maskData.push({ q, k, isOn, id: `cell-${q}-${k}`, color: cellColor });
            }
            // Labels
            const isClean = q < n_frames;
            const eff_idx = q % n_frames;
            const latexPrefix = (mode === 'training' && !isClean) ? "\\tilde" : "";
            const label_idx = (mode === 'training') ? eff_idx : q;
            labelData.push({ type: 'q', index: q, label: `$${latexPrefix}{Q}_{${label_idx + 1}}$` });
            labelData.push({ type: 'k', index: q, label: `$${latexPrefix}{K}_{${label_idx + 1}}$` });
        }
        
        // Draw Grid
        maskGridGroup.selectAll('rect.mask-cell')
            .data(maskData, d => d.id)
            .join(
                enter => enter.append('rect').style('opacity', 0),
                update => update,
                exit => exit.transition().duration(FADE_DURATION).style('opacity', 0).remove()
            )
            .attr('class', d => `mask-cell ${d.isOn ? 'on' : 'off'}`)
            .style('--cell-color', d => d.color)
            .transition().duration(FADE_DURATION)
            .attr('x', d => MASK_PADDING + MASK_LABEL_OFFSET + d.k * cellSize)
            .attr('y', d => MASK_PADDING + d.q * cellSize)
            .attr('width', cellSize).attr('height', cellSize)
            .style('opacity', 1);

        // Draw Tick Labels
        maskLabelsGroup.selectAll('.mask-tick')
            .data(labelData, d => `${d.type}-${d.index}`)
            .join(
                enter => enter.append('foreignObject').attr('class', 'mask-tick').style('opacity', 0),
                update => update,
                exit => exit.transition().duration(FADE_DURATION).style('opacity', 0).remove()
            )
            .html(d => `<span class="math-host-span">${d.label}</span>`)
            .attr('width', MASK_LABEL_OFFSET)
            .transition().duration(FADE_DURATION)
            .attr('height', cellSize)
            .attr('x', d => d.type === 'k' ? MASK_PADDING + MASK_LABEL_OFFSET + d.index * cellSize + cellSize / 2 - MASK_LABEL_OFFSET / 2 : MASK_PADDING)
            .attr('y', d => d.type === 'q' ? MASK_PADDING + d.index * cellSize : MASK_PADDING + MASK_GRID_SIZE + 10)
            .style('opacity', 1);

        renderKatex();
    }
    
    // --- Animation Functions ---
    function startTransitionAnimation(clickedNoisyIndex) {
        if (isAnimating) return;
        clearAllHighlights();
        isAnimating = true;
        isAnimationComplete = false;
        lastAnimatedIndex = clickedNoisyIndex;

        const n_frames = currentNFrames;
        const target_eff_idx = clickedNoisyIndex % n_frames;

        // --- Animate Main Graph ---
        const originalBoxData = boxesData.find(d => d.id === `q-${clickedNoisyIndex}`);
        const targetBoxData = boxesData.find(d => d.id === `q-${target_eff_idx}`);
        const translateX = targetBoxData.x - originalBoxData.x;

        linesGroup.selectAll('line.connection')
            .transition().duration(FADE_DURATION)
            .style('opacity', d => are_connected_inference_anim(d.from_q, d.to_k, n_frames, target_eff_idx) ? 0.8 : 0);
        
        d3.selectAll('#boxes-group rect, #labels-group foreignObject')
            .transition().duration(FADE_DURATION).delay(STAGGER_DELAY)
            .style('opacity', function(d) {
                const isClickedPair = d.index === clickedNoisyIndex;
                const isKeptClean = d.isClean && d.eff_idx <= target_eff_idx;
                d3.select(this).classed('faded-out', !(isClickedPair || isKeptClean));
                return (isClickedPair || isKeptClean) ? 1 : 0;
            });

        d3.selectAll(`#q-${clickedNoisyIndex}, #k-${clickedNoisyIndex}, #label-q-${clickedNoisyIndex}, #label-k-${clickedNoisyIndex}`)
            .transition().duration(MOVE_DURATION).delay(FADE_DURATION + STAGGER_DELAY)
            .attr('transform', `translate(${translateX}, 0)`)
            .on('end', () => {
                isAnimating = false;
                isAnimationComplete = true;
            });
        
        linesGroup.selectAll('line.connection')
            .filter(d => d.from_q === clickedNoisyIndex || d.to_k === clickedNoisyIndex)
            .transition().duration(MOVE_DURATION).delay(FADE_DURATION + STAGGER_DELAY)
            .attrTween('x1', function(d) {
                if (d.from_q !== clickedNoisyIndex) return null;
                const startX = parseFloat(d3.select(this).attr('x1'));
                return d3.interpolate(startX, startX + translateX);
            })
            .attrTween('x2', function(d) {
                if (d.to_k !== clickedNoisyIndex) return null;
                const startX = parseFloat(d3.select(this).attr('x2'));
                return d3.interpolate(startX, startX + translateX);
            });

        // --- Animate Mask ---
        drawMaskVisualization('inference', { n_frames, target_eff_idx });
    }

    function reverseTransitionAnimation() {
        if (isAnimating || lastAnimatedIndex === null) return;
        isAnimating = true;
        isAnimationComplete = false;
        
        const currentLastAnimatedIndex = lastAnimatedIndex;
        
        // Move animated elements back
        d3.selectAll(`#q-${currentLastAnimatedIndex}, #k-${currentLastAnimatedIndex}, #label-q-${currentLastAnimatedIndex}, #label-k-${currentLastAnimatedIndex}`)
            .transition().duration(MOVE_DURATION)
            .attr('transform', null); // Reset transform

        // Move their lines back
        linesGroup.selectAll('line.connection')
            .filter(d => d.from_q === currentLastAnimatedIndex || d.to_k === currentLastAnimatedIndex)
            .transition().duration(MOVE_DURATION)
            .attr('x1', d => boxesData.find(b => b.id === `q-${d.from_q}`).x + BOX_SIZE / 2)
            .attr('x2', d => boxesData.find(b => b.id === `k-${d.to_k}`).x + BOX_SIZE / 2);

        // Fade everything back in after the move starts
        d3.selectAll('#boxes-group rect, #labels-group foreignObject')
            .classed('faded-out', false)
            .transition().duration(FADE_DURATION).delay(MOVE_DURATION - 200)
            .style('opacity', 1);
            
        linesGroup.selectAll('line.connection')
            .transition().duration(FADE_DURATION).delay(MOVE_DURATION - 200)
            .style('opacity', 0.8)
            .on('end', (d, i, nodes) => {
                if (i === nodes.length - 1) {
                    isAnimating = false;
                    lastAnimatedIndex = null;
                    clearAllHighlights();
                    // Redraw to restore original colors and state
                    drawAttentionAnimation(currentNFrames);
                }
            });

        // Animate Mask Back
        drawMaskVisualization('training', { n_frames: currentNFrames });
    }

    // --- Event Handlers ---
    function handleBoxClick(event, d) {
        if (isAnimating) return;
        if (isAnimationComplete) {
            reverseTransitionAnimation();
            return;
        }
        if (d.type === 'q' && !d.isClean) {
            startTransitionAnimation(d.index);
        }
    }
    
    function handleMouseOver(event, d) {
        if (isAnimating || isAnimationComplete) return;
        
        linesGroup.selectAll('line.connection').classed('dimmed', true);
        const lineSelector = d.type === 'q' ? `[data-from-q="${d.index}"]` : `[data-to-k="${d.index}"]`;
        linesGroup.selectAll(lineSelector).classed('dimmed', false).classed('highlighted', true).raise();

        boxesGroup.selectAll('.box').classed('related-highlight', related_d => {
            if (d.type === 'q') return related_d.type === 'k' && are_connected_training(d.index, related_d.index, currentNFrames);
            else return related_d.type === 'q' && are_connected_training(related_d.index, d.index, currentNFrames);
        });

        maskGridGroup.selectAll('rect.mask-cell')
            .filter(cell_d => (d.type === 'q' && cell_d.q === d.index) || (d.type === 'k' && cell_d.k === d.index))
            .classed('highlighted', true);
    }

    function handleMouseOut() {
        if (isAnimating || isAnimationComplete) return;
        clearAllHighlights();
    }

    function clearAllHighlights() {
        linesGroup.selectAll('line.connection').classed('dimmed', false).classed('highlighted', false);
        boxesGroup.selectAll('.box').classed('related-highlight', false);
        maskGridGroup.selectAll('rect.mask-cell').classed('highlighted', false);
    }
    
    function handleSvgResetClick(event) {
        // Reset if the click is on the background of the animation, not its elements
        if (event.target === this && isAnimationComplete && !isAnimating) {
            reverseTransitionAnimation();
        }
    }
    
    // --- Math Rendering ---
    function renderKatex() {
        if (window.renderMathInElement) {
            renderMathInElement(svg.node(), {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '\\[', right: '\\]', display: true}
                ],
                throwOnError: false
            });
        }
    }

    // --- Initial Setup and Event Listeners ---
    animGroup.on('click', handleSvgResetClick);
    initializeVisualizations();
});