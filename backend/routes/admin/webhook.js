
const express = require('express');
const router = express.Router();
const connection = require('../../database/db');
const executeQuery = require('../../utils/executeQuery');
const fs = require('fs');
const { middleware } = require('../../middleware/middleware');




function buildFlowGraph(nodes, edges) {
    const graph = {};
    const nodeMap = {};

    nodes.forEach(node => {
        nodeMap[node.id] = node;
    });

    edges.forEach(edge => {
        if (!graph[edge.source]) {
            graph[edge.source] = [];
        }
        graph[edge.source].push({ target: edge.target });
    });

    return { graph, nodeMap };
}

router.post('/webhook', async (req, res) => {
    try {
        // console.log('=== NEW WEBHOOK REQUEST ===');
        const body = req.body;
        console.log("Raw incoming data:", JSON.stringify(body, null, 2));

        if (body?.object !== 'whatsapp_business_account') {
            return res.status(200).json({ messages: [] });
        }

        const message = body?.entry?.[0]?.changes?.[0]?.value?.messages?.[0];
        if (!message) {
            return res.status(200).json({ messages: [] });
        }

        const from = message.from;
        const flow_id = body?.entry?.[0]?.changes?.[0]?.value?.flow_id;
        const is_template = body?.entry?.[0]?.changes?.[0]?.value?.is_template || false;

        const parsedFlowId = parseInt(flow_id, 10);
        if (!parsedFlowId || isNaN(parsedFlowId)) {
            return res.status(400).json({ messages: [{ type: 'text', content: '❌ Invalid flow ID.' }] });
        }

        let msg = '';
        if (message.type === 'text') {
            msg = message.text?.body?.trim().toLowerCase();
        } else if (message.interactive) {
            msg = message.interactive.type === 'list_reply'
                ? message.interactive.list_reply.title.toLowerCase()
                : message.interactive.button_reply?.title.toLowerCase();
        }

        if (msg === 'restart') {
            await executeQuery('DELETE FROM user_node_progress WHERE phone_number = ?', [from]);
            await executeQuery('DELETE FROM user_answers WHERE phone_number = ?', [from]);
            return res.status(200).json({ messages: [{ type: 'text', content: '🔄 Flow reset. Type *hi* to begin again.' }] });
        }

        const [progress] = await executeQuery(
            'SELECT * FROM user_node_progress WHERE phone_number = ? ORDER BY id DESC LIMIT 1',
            [from]
        );

        let currentFlowId = progress?.flow_id || parsedFlowId;
        let nextNodeId = null;
        let messagesToSend = [];

        // ✅ Validate template existence
        if (is_template) {
            const [checkTemplate] = await executeQuery('SELECT id FROM template WHERE id = ?', [parsedFlowId]);
            if (!checkTemplate) {
                return res.status(404).json({ messages: [{ type: 'text', content: `❌ Template ID ${parsedFlowId} not found.` }] });
            }
        }

        // ✅ Load flow data
        const result = is_template
            ? await executeQuery('SELECT node AS nodes, edges FROM template WHERE id = ?', [parsedFlowId])
            : await executeQuery('SELECT nodes, edges FROM bots WHERE id = ?', [parsedFlowId]);

        if (!result || result.length === 0) {
            return res.status(404).json({ messages: [{ type: 'text', content: '❌ Flow not found.' }] });
        }

        const bot = result[0];
        const nodes = typeof bot.nodes === 'string' ? JSON.parse(bot.nodes) : bot.nodes;
        const edges = typeof bot.edges === 'string' ? JSON.parse(bot.edges) : bot.edges;
        const { graph, nodeMap } = buildFlowGraph(nodes, edges);

        let currentNodeId = progress?.current_node_id;

        // ✅ Start flow if first message
        if ((msg === 'hi' || msg === 'hello') && !progress) {
            const firstNode = nodes[0];

            await executeQuery(
                'INSERT INTO user_node_progress (phone_number, flow_id, current_node_id) VALUES (?, ?, ?)',
                [from, parsedFlowId, firstNode.id]
            );

            if (firstNode.type === 'ListButton') {
                messagesToSend.push({ type: 'list', label: firstNode.data.label, options: firstNode.data.targetValues });
            } else if (firstNode.type === 'ReplyButton') {
                messagesToSend.push({ type: 'buttons', label: firstNode.data.label, options: firstNode.data.targetValues });
            } else if (firstNode.type === 'LinkNode') {
                messagesToSend.push({ type: 'link', label: firstNode.data.linkText, url: firstNode.data.linkUrl });
            } else if (["GoogleSheetsNode", "VideoNode", "imageNode"].includes(firstNode.type)) {
                const mediaType = firstNode.type === 'GoogleSheetsNode' ? 'document' :
                    firstNode.type === 'VideoNode' ? 'video' : 'image';
                messagesToSend.push({ type: mediaType, url: firstNode.data?.fileUrl || firstNode.data?.fileName, caption: firstNode.data?.caption });
            } else {
                messagesToSend.push({ type: 'text', content: firstNode?.data?.label || '👋 Hello!' });
            }

            return res.status(200).json({ messages: messagesToSend });
        }

        // ✅ Continue flow
        if (!nodeMap[currentNodeId]) {
            currentNodeId = nodes[0].id;
            await executeQuery('UPDATE user_node_progress SET current_node_id = ? WHERE phone_number = ?', [currentNodeId, from]);
        }

        const currentNode = nodeMap[currentNodeId];

        if (message.type === 'text') {
            await executeQuery(
                'INSERT INTO user_answers (phone_number, flow_id, node_id, answer) VALUES (?, ?, ?, ?)',
                [from, parsedFlowId, currentNodeId, msg]
            );
        }

        const connections = graph[currentNodeId] || [];
        if (connections.length > 0) nextNodeId = connections[0].target;

        if (nextNodeId) {
            const nextNode = nodeMap[nextNodeId];

            await executeQuery('UPDATE user_node_progress SET current_node_id = ? WHERE phone_number = ?', [nextNodeId, from]);

            if (nextNode.type === 'ListButton') {
                messagesToSend.push({ type: 'list', label: nextNode.data.label, options: nextNode.data.targetValues });
            } else if (nextNode.type === 'ReplyButton') {
                messagesToSend.push({ type: 'buttons', label: nextNode.data.label, options: nextNode.data.targetValues });
            } else if (nextNode.type === 'LinkNode') {
                messagesToSend.push({ type: 'link', label: nextNode.data.linkText, url: nextNode.data.linkUrl });
            } else if (["GoogleSheetsNode", "VideoNode", "imageNode"].includes(nextNode.type)) {
                const mediaType = nextNode.type === 'GoogleSheetsNode' ? 'document' :
                    nextNode.type === 'VideoNode' ? 'video' : 'image';
                messagesToSend.push({ type: mediaType, url: nextNode.data?.fileUrl || nextNode.data?.fileName, caption: nextNode.data?.caption });
            } else {
                messagesToSend.push({ type: 'text', content: nextNode?.data?.label || '...' });
            }

        } else {
            messagesToSend.push({ type: 'text', content: '✅ Flow complete. Type *restart* to try again.' });
            await executeQuery('DELETE FROM user_node_progress WHERE phone_number = ?', [from]);
        }

        return res.status(200).json({ messages: messagesToSend });

    } catch (error) {
        console.error('❌ Error in webhook handler:', error);
        return res.status(500).json({ messages: [{ type: 'text', content: '❌ Internal Server Error' }] });
    }
});

router.get('/webhook-data', (req, res) => {
    console.log("webhook is active")
    res.send("hello")
})
module.exports = router;