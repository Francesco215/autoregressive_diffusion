document.addEventListener('DOMContentLoaded', function() {
    // --- D3 Visualization Script for Oniris VAE ---

    const svg = d3.select('#VAE-svg');
    const toggleButton = d3.select('#vae-toggle-button');
    let isGroupCausal = true;

    // --- Constants ---
    const BOX_SIZE = 28;
    const BOX_SPACING = 12;
    const PADDING = 20;
    const N_FRAMES = 16;
    const E1_PARAMS = { KERNEL_T: 6, GROUP_SIZE: 4 };
    const E3_PARAMS = { KERNEL_T: 3, GROUP_SIZE: 2 };
    const E5_PARAMS = { KERNEL_T: 2, GROUP_SIZE: 1 };
    
    // --- Helper Functions ---
    function getNodeColor(groupIndex) {
        const palette = [`hsl(210, 70%, 85%)`, `hsl(270, 70%, 85%)`, `hsl(30, 80%, 85%)`, `hsl(120, 70%, 85%)`];
        return palette[groupIndex % palette.length];
    }
    
    function getLinkColor(targetNode) {
        const palette = ['#3B82F6', '#A855F7', '#F97316', '#22C55E'];
        return palette[targetNode.group % palette.length];
    }

    function are_connected_group_causal(t_in, t_out, G, kt) {
        const i_out = Math.floor((t_out - 1) / G);
        const group_start = i_out * G + 1;
        const lookback = Math.max(0, kt - G);
        const start_in = Math.max(1, group_start - lookback);
        const end_in = group_start + G - 1;
        return t_in >= start_in && t_in <= end_in;
    }

    function are_connected_causal(t_in, t_out, kt) {
        return t_in <= t_out && (t_out - t_in) < kt;
    }

    // --- Main Drawing Function ---
    function drawVisualization() {
        svg.html('');
        
        const colSpacings = [50, 45, 33, 23, 50, 50, 23, 33, 45, 50].reverse();
        const layerX = {};
        let currentX = PADDING;
        const layerNames = ['d0', 'd1', 'd2', 'd3', 'd4', 'e5', 'e4', 'e3', 'e2', 'e1', 'e0'].reverse();
        layerNames.forEach((name, i) => {
            layerX[name] = currentX;
            if(i < colSpacings.length) currentX += BOX_SIZE + colSpacings[i];
        });

        const svgHeight = PADDING * 2 + N_FRAMES * BOX_SIZE + (N_FRAMES - 1) * BOX_SPACING;
        const svgWidth = currentX + BOX_SIZE + PADDING;
        svg.attr('viewBox', `0 0 ${svgWidth} ${svgHeight}`);

        let nodes = [];
        const addNodes = (count, type, group_size, x_pos, y_provider) => {
            for (let i = 1; i <= count; i++) {
                nodes.push({ id: `${type}-${i}`, type, index: i, group: Math.floor((i-1)/group_size), x: x_pos, y: y_provider(i)});
            }
        };
        
        addNodes(N_FRAMES, 'e0', E1_PARAMS.GROUP_SIZE, layerX.e0, i => PADDING + (i - 1) * (BOX_SIZE + BOX_SPACING));
        addNodes(N_FRAMES, 'e1', E1_PARAMS.GROUP_SIZE, layerX.e1, i => nodes.find(n => n.id === `e0-${i}`).y);
        addNodes(N_FRAMES / 2, 'e2', 2, layerX.e2, i => (nodes.find(n => n.id === `e1-${2 * i - 1}`).y + nodes.find(n => n.id === `e1-${2 * i}`).y) / 2);
        addNodes(N_FRAMES / 2, 'e3', E3_PARAMS.GROUP_SIZE, layerX.e3, i => nodes.find(n => n.id === `e2-${i}`).y);
        addNodes(N_FRAMES / 4, 'e4', 1, layerX.e4, i => (nodes.find(n => n.id === `e3-${2 * i - 1}`).y + nodes.find(n => n.id === `e3-${2 * i}`).y) / 2);
        addNodes(N_FRAMES / 4, 'e5', E5_PARAMS.GROUP_SIZE, layerX.e5, i => nodes.find(n => n.id === `e4-${i}`).y);
        
        ['e4', 'e3', 'e2', 'e1', 'e0'].forEach(encoderLayer => {
            const decoderLayer = encoderLayer.replace('e', 'd');
            const encoderNodes = nodes.filter(n => n.type === encoderLayer);
            encoderNodes.forEach(en => {
                nodes.push({ ...en, id: en.id.replace('e', 'd'), type: decoderLayer, x: layerX[decoderLayer] });
            });
        });

        const linkDefs = [
            { from: 'e0', to: 'e1', count: 16, params: E1_PARAMS, type: 'conv' },
            { from: 'e1', to: 'e2', count: 8, type: 'downsample' },
            { from: 'e2', to: 'e3', count: 8, params: E3_PARAMS, type: 'conv' },
            { from: 'e3', to: 'e4', count: 4, type: 'downsample' },
            { from: 'e4', to: 'e5', count: 4, params: E5_PARAMS, type: 'conv' },
            { from: 'e5', to: 'd4', count: 4, params: E5_PARAMS, type: 'conv' },
            { from: 'd4', to: 'd3', count: 4, type: 'upsample' },
            { from: 'd3', to: 'd2', count: 8, params: E3_PARAMS, type: 'conv' },
            { from: 'd2', to: 'd1', count: 8, type: 'upsample' },
            { from: 'd1', to: 'd0', count: 16, params: E1_PARAMS, type: 'conv' },
        ];
        const allLinksFlat = linkDefs.flatMap(def => {
            const links = [];
            if (def.type === 'conv') {
                const { KERNEL_T, GROUP_SIZE } = def.params;
                const effective_kt = KERNEL_T - GROUP_SIZE + 1;
                const rule = isGroupCausal ? are_connected_group_causal : are_connected_causal;
                const params = isGroupCausal ? [GROUP_SIZE, KERNEL_T] : [effective_kt];
                for (let t_out = 1; t_out <= def.count; t_out++) {
                    for (let t_in = 1; t_in <= def.count; t_in++) {
                        if (rule(t_in, t_out, ...params)) links.push({ sourceId: `${def.from}-${t_in}`, targetId: `${def.to}-${t_out}` });
                    }
                }
            } else if (def.type === 'downsample') {
                for (let i = 1; i <= def.count; i++) links.push({ sourceId: `${def.from}-${2*i-1}`, targetId: `${def.to}-${i}` }, { sourceId: `${def.from}-${2*i}`, targetId: `${def.to}-${i}` });
            } else if (def.type === 'upsample') {
                for (let i = 1; i <= def.count; i++) links.push({ sourceId: `${def.from}-${i}`, targetId: `${def.to}-${2*i-1}` }, { sourceId: `${def.from}-${i}`, targetId: `${def.to}-${2*i}` });
            }
            return links;
        });

        const allLinks = allLinksFlat.map(l => ({...l, source: nodes.find(n=>n.id===l.sourceId), target: nodes.find(n=>n.id===l.targetId)}));
        
        const decorationsGroup = svg.append('g');
        const linksGroup = svg.append('g');
        const boxesGroup = svg.append('g');

        linksGroup.selectAll('line.connection')
            .data(allLinks, d => `${d.sourceId}-${d.targetId}`).join('line')
            .attr('class', 'connection')
            .attr('x1', d => d.source.x + BOX_SIZE).attr('y1', d => d.source.y + BOX_SIZE / 2)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y + BOX_SIZE / 2);

        boxesGroup.selectAll('rect.box')
            .data(nodes, d => d.id).join('rect')
            .attr('class', 'box').attr('id', d => d.id)
            .attr('x', d => d.x).attr('y', d => d.y)
            .attr('width', BOX_SIZE).attr('height', BOX_SIZE).attr('rx', 4).attr('ry', 4)
            .style('fill', d => getNodeColor(d.group));
        
        linksGroup.selectAll('line').style('stroke', d => getLinkColor(d.target));

        const e5Nodes = nodes.filter(n => n.type === 'e5');
        const latentBoxPadding = 8;
        decorationsGroup.append('rect')
            .attr('x', e5Nodes[0].x - latentBoxPadding)
            .attr('y', e5Nodes[0].y - latentBoxPadding)
            .attr('width', BOX_SIZE + latentBoxPadding * 2)
            .attr('height', (e5Nodes[3].y + BOX_SIZE) - e5Nodes[0].y + latentBoxPadding * 2)
            .attr('rx', 4).attr('ry', 4)
            .attr('fill', 'none').attr('stroke', '#9ca3af').attr('stroke-width', 1.5).attr('stroke-dasharray', '6 4');

        boxesGroup.selectAll('rect.box').on('mouseover', function(event, d) {
            svg.selectAll('rect.box, line.connection').classed('dimmed', true);
            const pathIds = new Set();
            
            let queueDown = [d]; let visitedDown = new Set([d.id]);
            while(queueDown.length > 0) {
                const current = queueDown.shift(); pathIds.add(current.id);
                allLinks.forEach(link => {
                    if (link.source.id === current.id && !visitedDown.has(link.target.id)) {
                        visitedDown.add(link.target.id); queueDown.push(link.target);
                    }
                });
            }
            
            let queueUp = [d]; let visitedUp = new Set([d.id]);
             while(queueUp.length > 0) {
                const current = queueUp.shift(); pathIds.add(current.id);
                allLinks.forEach(link => {
                    if (link.target.id === current.id && !visitedUp.has(link.source.id)) {
                        visitedUp.add(link.source.id); queueUp.push(link.source);
                    }
                });
            }
            
            const pathNodes = svg.selectAll('rect.box').filter(node => pathIds.has(node.id));
            pathNodes.classed('dimmed', false);
            pathNodes.filter(node => node.type === 'e0' || node.type === 'd0').classed('highlighted', true);

            linksGroup.selectAll('line.connection').filter(l => pathIds.has(l.source.id) && pathIds.has(l.target.id))
                .classed('dimmed', false).classed('cross-group', l => l.source.type === 'e0' && l.source.group !== l.target.group);
        });

        boxesGroup.selectAll('rect.box').on('mouseleave', () => svg.selectAll('.dimmed, .highlighted, .cross-group').classed('dimmed highlighted cross-group', false));
    }

    // --- Initializer ---
    toggleButton.on('click', (event) => { event.preventDefault(); isGroupCausal = !isGroupCausal; toggleButton.text(isGroupCausal ? 'Switch to Causal' : 'Switch to Group-Causal'); drawVisualization(); });
    drawVisualization();
});