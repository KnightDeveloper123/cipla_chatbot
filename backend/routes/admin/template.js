const express = require('express');
const router = express.Router();
const connection = require('../../database/db');
const executeQuery = require('../../utils/executeQuery');
const fs = require('fs');
const { middleware } = require('../../middleware/middleware');

const multer = require('multer');
const path = require('path');


const sheetStorage = multer.diskStorage({
    destination: (req, file, cb) => {
        const sheetPath = path.join(__dirname, '../../uploadFiles');
        if (!fs.existsSync(sheetPath)) {
            fs.mkdirSync(sheetPath, { recursive: true });
        }
        cb(null, sheetPath);
    },
    filename: (req, file, cb) => {
        const { fileName } = req.query;
        if (!fileName) return cb(new Error('fileName is required'), null);
        cb(null, fileName + path.extname(file.originalname));
    }
});
const uploadSheet = multer({ storage: sheetStorage });


router.post('/upload-sheet', uploadSheet.single('file'), (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

        const originalName = req.query.fileName;
        const baseName = path.basename(originalName, path.extname(originalName));
        const extension = path.extname(req.file.originalname);
        const fileName = `${Date.now()}-${baseName}${extension}`;

        const finalPath = path.join(__dirname, "../../uploadFiles", fileName);

        fs.rename(req.file.path, finalPath, (err) => {
            if (err) return res.status(500).json({ error: "File rename failed" });
            return res.json({ success: true, fileName });
        });

    } catch (error) {
        console.error("GoogleSheet upload error:", error);
        res.status(500).json({ error: "Server Error" });
    }
});


router.post('/add', middleware, (req, res) => {
    const { industry, category, description } = req.body;
    // console.log(req.body)

    const sql = `INSERT INTO template(industry, category,description) VALUES (?, ?,?)`;
    connection.query(sql, [industry, category, description], (err, result) => {
        if (err) {
            console.error('Error saving flow:', err);
            return res.status(500).json({ message: 'Database error' });
        }
        res.status(200).json({ success: 'successfully', flowId: result.insertId });
    });
});

router.get('/get_all_template', middleware, async (req, res) => {
    try {
        const { admin_id } = req.query
        const data = await executeQuery(`select * from  template where status=0 and admin_id=${admin_id} order by id desc`)
        return res.json({ data })
    } catch (error) {
        console.log(error);
        return res.status(500).json({ error: "Internal server error!" })
    }
})

// user side ,
router.get('/get_all_templates', middleware, async (req, res) => {
    try {

        const data = await executeQuery(`select * from  template where status=0  order by id desc`)
        return res.json({ data })
    } catch (error) {
        console.log(error);
        return res.status(500).json({ error: "Internal server error!" })
    }
})


router.post('/create', (req, res) => {
    const { node, edges, template_id } = req.body;

    const sql = `UPDATE template SET node = ?, edges = ? WHERE id = ?`;

    connection.query(sql, [JSON.stringify(node), JSON.stringify(edges), template_id], (err, result) => {
        if (err) {
            console.error('Error saving flow:', err);
            return res.status(500).json({ message: 'Database error' });
        }
        res.status(200).json({ success: 'successfully' });
    });
});

router.get('/getbyid', async (req, res) => {
    try {
        const { id } = req.query;
        const data = await executeQuery(`SELECT * FROM template where id=${id}`)
        res.json({ data: data[0] });
    } catch (error) {
        console.log(error)
        return res.status(500).json({ error: "Internal Server Error" })
    }
});


router.post("/update", (req, res) => {
    const { id, node, edges, } = req.body;

    const query = `
    UPDATE template
    SET node = ?, edges = ?
    WHERE id = ?
  `;

    connection.query(query, [node, edges, id], (err, result) => {
        if (err) {
            console.error("MySQL Error:", err);
            return res.status(500).json({ error: "Failed to update template" });
        }

        res.status(200).json({ success: "Template updated successfully", result });
    });
});


// router.post('/webhook', async (req, res) => {
//     try {
//         console.log('=== NEW WEBHOOK REQUEST ===');
//         const body = req.body;
//         console.log("Raw incoming data:", JSON.stringify(body, null, 2));

//         if (body?.object !== 'whatsapp_business_account') {
//             return res.status(200).json({ messages: [] });
//         }

//         const message = body?.entry?.[0]?.changes?.[0]?.value?.messages?.[0];
//         if (!message) {
//             return res.status(200).json({ messages: [] });
//         }

//         const from = message.from;
//         const flow_id = body?.entry?.[0]?.changes?.[0]?.value?.flow_id;
//         console.log("flow_id",flow_id)
//         const is_template = body?.entry?.[0]?.changes?.[0]?.value?.is_template || false;
//         let messagesToSend = [];

//         let msg = '';
//         if (message.type === 'text') {
//             msg = message.text?.body?.trim().toLowerCase();
//         } else if (message.interactive) {
//             msg = message.interactive.type === 'list_reply'
//                 ? message.interactive.list_reply.title.toLowerCase()
//                 : message.interactive.button_reply?.title.toLowerCase();
//         }

//         if (msg === 'restart') {
//             await executeQuery('DELETE FROM user_node_progress WHERE phone_number = ?', [from]);
//             await executeQuery('DELETE FROM user_answers WHERE phone_number = ?', [from]);
//             messagesToSend.push({ type: 'text', content: '🔄 Flow reset. Type *hi* to begin again.' });
//             return res.status(200).json({ messages: messagesToSend });
//         }

//         const [progress] = await executeQuery(
//             'SELECT * FROM user_node_progress WHERE phone_number = ? ORDER BY id DESC LIMIT 1',
//             [from]
//         );

//         let currentFlowId = progress?.flow_id || flow_id;
//         let nextNodeId = null;

//         // Load flow (from bots or template)
//         let result;
//         if ((msg === 'hi' || msg === 'hello') && !progress) {
//             if (!currentFlowId) {
//                 const [defaultBot] = await executeQuery('SELECT id FROM bots LIMIT 1');
//                 currentFlowId = defaultBot?.id;
//             }

//             result = is_template
//                 ? await executeQuery('SELECT node AS nodes, edges FROM template WHERE id = ?', [currentFlowId])
//                 : await executeQuery('SELECT nodes, edges FROM bots WHERE id = ?', [currentFlowId]);

//             if (!result || result.length === 0) {
//                 return res.status(404).json({ messages: [{ type: 'text', content: '❌ Flow not found.' }] });
//             }

//             const bot = result[0];
//             const nodes = typeof bot.nodes === 'string' ? JSON.parse(bot.nodes) : bot.nodes;
//             const edges = typeof bot.edges === 'string' ? JSON.parse(bot.edges) : bot.edges;
//             const { graph, nodeMap } = buildFlowGraph(nodes, edges);
//             const firstNode = nodes[0];

//             if (firstNode.type === 'ListButton') {
//                 messagesToSend.push({ type: 'list', label: firstNode.data.label, options: firstNode.data.targetValues });
//             } else if (firstNode.type === 'ReplyButton') {
//                 messagesToSend.push({ type: 'buttons', label: firstNode.data.label, options: firstNode.data.targetValues });
//             } else if (firstNode.type === 'LinkNode') {
//                 messagesToSend.push({ type: 'link', label: firstNode.data.linkText, url: firstNode.data.linkUrl });
//             } else if (["GoogleSheetsNode", "VideoNode", "imageNode"].includes(firstNode.type)) {
//                 const mediaType = firstNode.type === 'GoogleSheetsNode' ? 'document' :
//                     firstNode.type === 'VideoNode' ? 'video' : 'image';
//                 messagesToSend.push({ type: mediaType, url: firstNode.data?.fileUrl || firstNode.data?.fileName, caption: firstNode.data?.caption });
//             } else {
//                 messagesToSend.push({ type: 'text', content: firstNode?.data?.label || '👋 Hello!' });
//             }

//             await executeQuery(
//                 'INSERT INTO user_node_progress (phone_number, flow_id, current_node_id) VALUES (?, ?, ?)',
//                 [from, currentFlowId, firstNode.id]
//             );

//             return res.status(200).json({ messages: messagesToSend });
//         }

//         if (!currentFlowId) {
//             return res.status(400).json({ messages: [{ type: 'text', content: '❌ No flow ID provided' }] });
//         }

//         result = is_template
//             ? await executeQuery('SELECT node AS nodes, edges FROM template WHERE id = ?', [currentFlowId])
//             : await executeQuery('SELECT nodes, edges FROM bots WHERE id = ?', [currentFlowId]);

//         if (!result || result.length === 0) {
//             return res.status(404).json({ messages: [{ type: 'text', content: '❌ Flow not found.' }] });
//         }

//         const bot = result[0];
//         const nodes = typeof bot.nodes === 'string' ? JSON.parse(bot.nodes) : bot.nodes;
//         const edges = typeof bot.edges === 'string' ? JSON.parse(bot.edges) : bot.edges;
//         const { graph, nodeMap } = buildFlowGraph(nodes, edges);

//         let currentNodeId = progress?.current_node_id;
//         if (!nodeMap[currentNodeId]) {
//             currentNodeId = nodes[0].id;
//             await executeQuery('UPDATE user_node_progress SET current_node_id = ? WHERE phone_number = ?', [currentNodeId, from]);
//         }

//         const currentNode = nodeMap[currentNodeId];

//         if (message.type === 'text') {
//             await executeQuery(
//                 'INSERT INTO user_answers (phone_number, flow_id, node_id, answer) VALUES (?, ?, ?, ?)',
//                 [from, currentFlowId, currentNodeId, msg]
//             );
//         }

//         const connections = graph[currentNodeId] || [];
//         if (connections.length > 0) nextNodeId = connections[0].target;

//         if (nextNodeId) {
//             const nextNode = nodeMap[nextNodeId];
//             await executeQuery('UPDATE user_node_progress SET current_node_id = ? WHERE phone_number = ?', [nextNodeId, from]);

//             if (nextNode.type === 'ListButton') {
//                 messagesToSend.push({ type: 'list', label: nextNode.data.label, options: nextNode.data.targetValues });
//             } else if (nextNode.type === 'ReplyButton') {
//                 messagesToSend.push({ type: 'buttons', label: nextNode.data.label, options: nextNode.data.targetValues });
//             } else if (nextNode.type === 'LinkNode') {
//                 messagesToSend.push({ type: 'link', label: nextNode.data.linkText, url: nextNode.data.linkUrl });
//             } else if (["GoogleSheetsNode", "VideoNode", "imageNode"].includes(nextNode.type)) {
//                 const mediaType = nextNode.type === 'GoogleSheetsNode' ? 'document' :
//                     nextNode.type === 'VideoNode' ? 'video' : 'image';
//                 messagesToSend.push({ type: mediaType, url: nextNode.data?.fileUrl || nextNode.data?.fileName, caption: nextNode.data?.caption });
//             } else {
//                 messagesToSend.push({ type: 'text', content: nextNode?.data?.label || '...' });
//             }
//         } else {
//             messagesToSend.push({ type: 'text', content: '✅ Flow complete. Type *restart* to try again.' });
//             await executeQuery('DELETE FROM user_node_progress WHERE phone_number = ?', [from]);
//         }

//         return res.status(200).json({ messages: messagesToSend });
//     } catch (error) {
//         console.error('❌ Error in webhook handler:', error);
//         return res.status(500).json({ messages: [{ type: 'text', content: '❌ Internal Server Error' }] });
//     }
// });




module.exports = router;