const express = require('express');
const router = express.Router();
const connection = require('../../database/db');
const executeQuery = require('../../utils/executeQuery');
const csv = require('csv-parser');
const fs = require('fs');
const { middleware } = require('../../middleware/middleware');
const multer = require('multer')
const path = require('path')

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, path.join(__dirname, '../../contactList'));
    },
    filename: (req, file, cb) => {
        const { fileName } = req.query;
        if (!fileName) {
            return cb(new Error('fileName is required'), null);
        }
        cb(null, fileName + path.extname(file.originalname));
    },
});

const upload = multer({
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 },
});


// router.post('/upload', middleware, upload.single('file'), async (req, res) => {
//     try {
//         const { fileName } = req.query;
//         const { admin_id } = req.body

//         if (!req.file || !fileName) {
//             return res.status(400).json({ error: 'File and fileName are required.' });
//         }

//         const fileExtension = path.extname(req.file.originalname);
//         const file_name = fileName + fileExtension;

//         connection.query('INSERT INTO contacts (name,contact_name,email,phone,admin_id) VALUES (?, ?, ?, ?, ?)', [file_name,admin_id], (err, data) => {
//             if (err) {
//                 console.log(err);
//                 return res.status(400).json({ error: "Something went wrong" })
//             }
//             return res.status(200).json({
//                 success: 'File uploaded successfully.',
//                 data
//             });

//         });

//     } catch (error) {
//         console.error('/upload Error:', error.message);
//         return res.status(500).json({ error: 'Internal Server Error.' });
//     }
// });

router.post('/upload', middleware, upload.single('file'), async (req, res) => {
    try {
        const { admin_id, fileName } = req.body;

        if (!req.file || !admin_id || !fileName) {
            return res.status(400).json({ error: 'File, admin_id, and fileName are required.' });
        }

        const fileExtension = path.extname(req.file.originalname);
        const file_name = `${fileName}${fileExtension}`;
        const results = [];

        fs.createReadStream(req.file.path)
            .pipe(csv())
            .on('data', (row) => {
                console.log(row)
                const { contact_name, phone, email } = row;
                if (contact_name && phone && email) {
                    results.push([file_name, contact_name, phone, email, admin_id]);
                }
            })
            .on('end', () => {
                if (results.length === 0) {
                    // console.log(results)
                    return res.status(400).json({ error: 'No valid data found in CSV.' });
                }

                const sql = 'INSERT INTO contacts (name, contact_name, phone, email, admin_id) VALUES ?';
                connection.query(sql, [results], (err, data) => {
                    if (err) {
                        console.error('Insert error:', err);
                        return res.status(500).json({ error: 'Error inserting contacts.' });
                    }

                    return res.status(200).json({
                        success: 'Contacts uploaded successfully.',
                        inserted: data.affectedRows
                    });
                });
            });

    } catch (error) {
        console.error('/upload Error:', error);
        return res.status(500).json({ error: 'Internal Server Error.' });
    }
});
router.post("/delete", middleware, async (req, res) => {
    try {
        const { contact_id } = req.query;

        if (!contact_id) {
            return res.status(400).json({ error: "Query is required" })
        }
        const [contacts] = await executeQuery(`SELECT name FROM contacts WHERE id = ${contact_id} and status=0`);

        if (!contacts) {
            return res.status(404).json({ error: "contacts not found" });
        }
        const fileName = contacts.name;
        const filePath = path.join(__dirname, '../../contactList', fileName);
        console.log(fileName, filePath);

        try {
            await fs.promises.unlink(filePath);
            connection.query(`update contacts set status=1 where id=?`, [contact_id], (err, data) => {
                if (err) {
                    console.log(err);
                    return res.status(400).json({ error: "Something went wrong" })
                }
                return res.json({ success: "File deleted successfully.", data });
            })
        } catch (unlinkError) {
            if (unlinkError.code === 'ENOENT') {
                return res.status(404).json({ error: "File not found" });
            }
            return res.status(500).json({ error: "Failed to delete file" });
        }
    } catch (error) {
        console.error("Error in /delete:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.get("/getAllcontacts", async (req, res) => {
    try {
        const {admin_id}=req.query;
      
        connection.query(`select * from contacts where status=0 AND admin_id=${admin_id}`, (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            // console.log(data)
            return res.json({ success: "success", data })
        })
    } catch (error) {
        console.error("Error in /getAllcontacts:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

module.exports = router;