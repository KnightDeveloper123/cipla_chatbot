const express = require('express');
const connection = require('../../database/db');
const executeQuery = require('../../utils/executeQuery');
const { middleware } = require('../../middleware/middleware');
const { Parser } = require('json2csv');
const { format } = require('fast-csv');
const fs = require('fs');
const PDFDocument = require('pdfkit');
const router = express.Router();
const axios = require('axios');
const path = require('path');
const multer = require('multer');

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.join(__dirname, '../../banner'));
  },
  filename: (req, file, cb) => {
    const { fileName } = req.query;

    cb(null, fileName + path.extname(file.originalname));
  },
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 },
});

router.post("/add", middleware, upload.single('file'), async (req, res) => {
  try {
    const { fileName } = req.body;
    const {
      channel_name,
      campaign_name,
      template_name,
      template_type,
      template_lang,
      header,
      body,
      admin_id,
      sector_id,
      to,
      banner,
      cta_label,
      cta_link
      //   bot_type
    } = req.body;

    // console.log(req.body);

    const insertQuery = `
  INSERT INTO campaign 
  (channel_name, campaign_name, template_name, template_type, template_lang, header, body, is_status, admin_id, sector_id, \`to\`, banner, cta_label, cta_link)
  VALUES (?, ?, ?, ?, ?, ?, ?, "Pending", ?, ?, ?, ?, ?, ?);
`;

    const values = [
      channel_name ?? null,
      campaign_name ?? null,
      template_name ?? null,
      template_type ?? null,
      template_lang ?? null,
      header ?? null,
      body ?? null,
      admin_id ?? null,
      sector_id ?? null,
      to ?? null,
      fileName ?? null,       // this will go to 'banner' column
      cta_label ?? null,
      cta_link ?? null
    ];


    connection.execute(insertQuery, values, (err, data) => {
      if (err) {
        console.log(err);
        return res.status(400).json({ error: "Something went wrong" });
      }
      else {
        const flowId = data.insertId;

        const insertquery = `INSERT INTO interactions_log ( bot_id, campaign_id, interaction_type) values(?,?,?)`;
        connection.query(insertquery, [null, flowId, "click"], (err, result) => {
          if (err) {
            console.error('Error saving flow:', err);
            return res.status(500).json({ message: 'Database error' });
          } else {

            return res.json({ success: "Campaign and interaction  Added successfully", data, flowId });
          }
        });

      }
    })

  } catch (error) {
    console.log("/add: ", error.message);
    return res.status(500).json({ error: "Internal Server Error." });
  }
});

const db = connection.promise();

async function sendWhatsAppMessage({ phone, message }) {
  // console.log(message)
  const token = process.env.WHATSAPP_TOKEN;
  const phoneNumberId = process.env.PHONE_NUMBER_ID;

  const url = `https://graph.facebook.com/v17.0/${phoneNumberId}/messages`;

  try {
    const response = await axios.post(
      url,
      {
        messaging_product: 'whatsapp',
        to: phone,
        type: 'text',
        text: { body: message },
      },
      {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      }
    );

    // console.log('Message sent:', response.data);
    return response.data;
  } catch (error) {
    console.error('Failed to send message:', error.response?.data || error.message);
    throw error;
  }
}

async function sendMessageToUser({ campaignIds = [], contactIds = [] }) {
  if (!campaignIds.length || !contactIds.length) {
    throw new Error('Please provide both campaignIds and contactIds');
  }

  // console.log(campaignIds);
  // console.log(contactIds);

  // ✅ use await db.query() instead
  const [campaigns] = await db.query(
    `SELECT id, channel_name, campaign_name, body, message_content
     FROM campaign WHERE id IN (?)`,
    [campaignIds]
  );

  const [contacts] = await db.query(
    `SELECT id, contact_name, phone, email
     FROM contacts WHERE id IN (?)`,
    [contactIds]
  );

  // console.log(campaigns); // should now show an array

  const results = [];

  for (const campaign of campaigns) {
    for (const contact of contacts) {
      // console.log("campaigan"+ campaign )
      // console.log("body"+ campaign.body )
      try {
        await sendWhatsAppMessage({
          phone: contact.phone,
          message: campaign.body
        });

        await db.query(
          `INSERT INTO messages_log
           (campaign_id, contact_id, status, sent_at)
           VALUES (?, ?, 'sent', NOW())`,
          [campaign.id, contact.id]
        );

        results.push({ contactId: contact.id, campaignId: campaign.id, status: 'Sent' });
      } catch (err) {
        results.push({ contactId: contact.id, campaignId: campaign.id, status: 'Failed' });
      }
    }
  }

  return results;
}


router.post('/send-campaign', async (req, res) => {
  const { campaign_ids, contact_ids } = req.body;

  if (!campaign_ids?.length || !contact_ids?.length) {
    return res.status(400).json({ message: 'Please select campaigns and contacts' });
  }
  // console.log("hi"+campaign_ids)
  // console.log("hello"+contact_ids)
  try {
    const result = await sendMessageToUser({
      campaignIds: campaign_ids,
      contactIds: contact_ids,
    });

    res.json({ message: 'Campaign(s) sent successfully', result });
  } catch (error) {
    console.error('Error sending campaign:', error);
    res.status(500).json({ message: 'Failed to send campaign' });
  }
});



router.post("/update", middleware, async (req, res) => {
  try {
    const {
      channel_name, campaign_name,
      template_name, template_type, template_lang, header, body, campaign_id } = req.body;

    let updateFields = [];
    let updateValues = [];

    if (channel_name) {
      updateFields.push("channel_name = ?");
      updateValues.push(channel_name);
    }
    if (campaign_name) {
      updateFields.push("campaign_name = ?");
      updateValues.push(campaign_name);
    }
    // if (message_content) {
    //   updateFields.push("message_content = ?");
    //   updateValues.push(message_content);
    // }
    // if (sector) {
    // updateFields.push("sector = ?");
    // updateValues.push(sector);
    // } 
    if (template_name) {
      updateFields.push("template_name = ?");
      updateValues.push(template_name);
    }

    if (template_type) {
      updateFields.push("template_type = ?");
      updateValues.push(template_type);
    }
    if (template_lang) {
      updateFields.push("template_lang = ?");
      updateValues.push(template_lang);
    }

    if (header) {
      updateFields.push("header = ?");
      updateValues.push(header);
    }
    if (body) {
      updateFields.push("body = ?");
      updateValues.push(body);
    }

    if (updateFields.length === 0) {
      return res.status(400).json({ error: "No fields to update" });
    }

    const updateSql = `UPDATE campaign SET ${updateFields.join(', ')} WHERE id = ?`;
    updateValues.push(campaign_id);

    connection.query(updateSql, updateValues, (updateErr) => {
      if (updateErr) {
        console.error(updateErr);
        return res.status(400).json({ error: "Error updating Campaign" });
      }
      return res.send({ success: "Campaign updated successfully" });

    });
  } catch (error) {
    console.error("Error in /update :", error.message);
    return res.status(500).json({ error: "Internal Server Error" });
  }
});

router.post("/delete", middleware, async (req, res) => {
  try {
    const { user_id } = req.body;

    const query = `UPDATE campaign SET status=1, updated_at = CURRENT_TIMESTAMP WHERE id = ?`;

    connection.execute(query, [user_id], (err, data) => {
      if (err) {
        console.log(err);
        return res.status(400).json({ error: "Something went wrong" })
      }
      if (data.affectedRows === 0) {
        return res.status(404).json({ error: "Record not found" });
      }
      return res.json({ success: "Campaign deleted", data })
    });
  } catch (error) {
    console.error("Error in /updateUser:", error.message);
    return res.status(500).json({ error: "Internal Server Error" });
  }
});

router.get('/getCampaign', middleware, async (req, res) => {
  try {
    const { user_id } = req.query;
    const data = await executeQuery(`select campaign.*, s.name as sname from campaign
        left join sector as s on campaign.sector_id=s.id
        where campaign.id=${user_id}`)
    return res.json({ data: data[0] })
  } catch (error) {
    // console.log(error)
    return res.status(500).json({ error: "Internal Server Error" })
  }
});

router.get('/getAllCampaign', middleware, async (req, res) => {
  try {
    const { admin_id } = req.query;
    // console.log("id",admin_id)

    const data = await executeQuery(`SELECT 
      c.*, 
      s.name AS sname,ml.status as ml_status,
      COUNT(ml.id) AS sent_count
    FROM campaign c
    LEFT JOIN sector s ON c.sector_id = s.id
    LEFT JOIN messages_log ml ON ml.campaign_id = c.id AND ml.status = 'Sent'
    WHERE c.admin_id = ${admin_id}
    GROUP BY c.id
    ORDER BY c.id DESC;
    `)
    return res.json({ data })
  } catch (error) {
    // console.log(error)
    return res.status(500).json({ error: "Internal Server Error" })
  }
});

router.post('/track-user-campaign', async (req, res) => {
  const { campaign_id, user_id } = req.body;
  // console.log(req.body)

  if (!campaign_id || !user_id) {
    return res.status(400).json({ success: false, message: 'botId and userId are required' });
  }

  try {
    // Prevent duplicate entries (optional - works only if UNIQUE(bot_id, user_id) is set)
    await executeQuery(`
      INSERT  INTO campaign_users ( campaign_id,user_id ) VALUES (?, ?)
    `, [campaign_id, user_id]);

    return res.status(201).json({ success: true, message: 'User-bot tracked successfully' });
  } catch (err) {
    console.error('Error inserting into bot_users:', err);
    return res.status(500).json({ success: false, message: 'Internal Server Error' });
  }
});

router.get('/export', async (req, res) => {
  const { status } = req.query;

  let query = 'SELECT * FROM campaign';
  const values = [];

  if (status !== undefined) {
    query += ' WHERE status = ?';
    values.push(status);
  }

  const [rows] = await executeQuery(query, values);
  const parser = new Parser();
  const csv = parser.parse(rows);

  res.header('Content-Type', 'text/csv');
  res.attachment('campaigns.csv');
  res.send(csv);
});


// for export csv
router.get('/export/campaigns/csv', async (req, res) => {
  try {
    const campaigns = await executeQuery(`
      SELECT id, campaign_name, channel_name, is_status, created_at
      FROM campaign
      ORDER BY created_at DESC
    `);

    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename=campaigns.csv');

    const csvStream = format({ headers: true });
    csvStream.pipe(res);
    campaigns.forEach(row => csvStream.write(row));
    csvStream.end();

  } catch (err) {
    console.error('CSV Export Error:', err);
    res.status(500).json({ success: false, message: 'Error exporting CSV' });
  }
});


// for singal campaign 
router.get('/export/campaigns/csv/:id', async (req, res) => {
  const campaignId = req.params.id;

  try {
    const result = await executeQuery(`
      SELECT id, campaign_name,template_name,template_type, channel_name, is_status, created_at
      FROM campaign
      WHERE id = ?
    `, [campaignId]);

    if (!result.length) {
      return res.status(404).json({ success: false, message: 'Campaign not found' });
    }

    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename=campaign_${campaignId}.csv`);

    const csvStream = format({ headers: true });
    csvStream.pipe(res);
    csvStream.write(result[0]); // 👈 export only the single row
    csvStream.end();

  } catch (err) {
    console.error('CSV Export Error:', err);
    res.status(500).json({ success: false, message: 'Error exporting CSV' });
  }
});

//export pdf 
router.get('/export/campaigns/pdf', async (req, res) => {
  try {
    const campaigns = await executeQuery(`
      SELECT id, campaign_name, channel_name, is_status, created_at
      FROM campaign
      ORDER BY created_at DESC
    `);

    const doc = new PDFDocument({ margin: 30 });
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename=campaigns.pdf');
    doc.pipe(res);

    doc.fontSize(16).text('Campaign Report', { align: 'center' });
    doc.moveDown();

    campaigns.forEach(c => {
      doc.fontSize(12).text(`Name: ${c.campaign_name}`);
      doc.text(`Channel: ${c.channel_name}`);
      doc.text(`Status: ${c.is_status}`);
      doc.text(`Created At: ${new Date(c.created_at).toLocaleString()}`);
      doc.moveDown();
    });

    doc.end();

  } catch (err) {
    console.error('PDF Export Error:', err);
    res.status(500).json({ success: false, message: 'Error exporting PDF' });
  }
});


module.exports = router;