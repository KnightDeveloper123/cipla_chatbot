const express = require('express');
const router = express.Router();
const { middleware } = require('../../middleware/middleware')
const connection = require('../../database/db');
const executeQuery = require('../../utils/executeQuery');
const multer = require('multer');
const { BOT_URL } = require('../../utils/variables');

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/community'); // make sure this folder exists
    },
    filename: function (req, file, cb) {
        const filename = file.originalname;
        cb(null, filename);
    }
});

const upload = multer({ storage: storage });

router.post('/add-question', middleware, (req, res) => {
    try {
        const { question, user_id } = req.body

        connection.query('insert into community_questions (question, user_id) values (?, ?);', [question, user_id], (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Someting went wrong" })
            }
            return res.json({ success: "Question added" })
        });

    } catch (error) {
        console.log(error);
        return res.status(500).json({ error: "Internal server error" })
    }
})


router.post('/update-question', middleware, (req, res) => {
    try {
        const { question, question_id } = req.body

        if (!question || !question_id) {
            return res.status(400).json({ error: "Invalid request" });
        }

        connection.query('update community_questions set question = ?, updated_on=NOW() where id = ?', [question, question_id], (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Someting went wrong" })
            }
            return res.json({ success: "Question updated" })
        });

    } catch (error) {
        console.log(error);
        return res.status(500).json({ error: "Internal server error" })
    }
})

router.post('/delete-question', middleware, (req, res) => {
    try {
        const { question_id } = req.body

        if (!question_id) {
            return res.status(400).json({ error: "Invalid request" });
        }

        connection.query(`update community_questions set status = '1', updated_on=NOW() where id = ?`, [question_id], (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Someting went wrong" })
            }
            return res.json({ success: "Question deleted" })
        });

    } catch (error) {
        console.log(error);
        return res.status(500).json({ error: "Internal server error" })
    }
})

router.get('/get-all-question', middleware, async (req, res) => {
    try {

        const questions = await executeQuery(`select cq.*, u.name from community_questions cq left join user u on cq.user_id=u.id where cq.status = '0' order by cq.created_at desc;`);
        const questionIds = questions?.map(elem => elem.id)

        const answers = await executeQuery(`select ca.*, u.name from community_answers ca left join user u on ca.user_id=u.id where ca.status = '0' and ca.question_id in (${questionIds?.join(',')}) order by ca.created_at desc;`)

        const data = questions?.map(elem => {
            const ans = answers?.filter(el => el.question_id === elem.id);
            const result = { ...elem, answers: ans }
            return result
        })
        return res.json({ success: "success", data })

    } catch (error) {
        console.log(error);
        return res.status(500).json({ error: "Internal server error" })
    }
})

router.post('/add-answer', middleware, upload.single('file'), (req, res) => {
    try {
        const { answer, question_id, user_id } = req.body;

        if (!answer || !question_id || !user_id) {
            return res.status(400).json({ error: "Missing required fields" });
        }

        connection.query(
            'INSERT INTO community_answers (answer, file, question_id, user_id) VALUES (?, ?, ?, ?)',
            [answer, req.file?.filename || null, question_id, user_id],
            (err, result) => {
                if (err) {
                    console.log(err);
                    return res.status(400).json({ error: "Something went wrong" });
                }
                return res.json({ success: "Answer added successfully" });
            }
        );
    } catch (error) {
        console.log(error);
        return res.status(500).json({ error: "Internal server error" });
    }
});

router.post('/update-answer', middleware, upload.single('file'), async (req, res) => {
    try {
        const { answer_id, ...rest } = req.body;

        if (!answer_id) {
            return res.status(400).json({ error: "Invalid request" });
        }

        const fields = Object.keys(rest);
        const values = Object.values(rest);

        if (fields.length === 0) {
            return res.status(400).json({ error: "No valid fields provided for update." });
        }

        if (req.file) {
            fields.push('file')
            values.push(req.file.filename)
        }

        const setClause = [...fields, 'updated_on'].map(field => field === 'updated_on' ? 'updated_on = now()' : `${field} = ?`).join(", ");
        values.push(answer_id);
        const query = `UPDATE community_answers SET ${setClause} WHERE id = ?`;

        connection.execute(query, values, async (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            if (data.affectedRows === 0) {
                return res.status(404).json({ error: "Record not found" });
            }

            return res.json({ success: "Answer updated", data })
        });

    } catch (error) {
        console.log(error)
        return res.status(500).json({ error: "Internal server error" });
    }
});


router.post('/delete-answer', middleware, (req, res) => {
    try {
        const { answer_id } = req.body;

        if (!answer_id) {
            return res.status(400).json({ error: "Answer ID is required" });
        }

        connection.query(
            `UPDATE community_answers SET status = '1', updated_on = NOW() WHERE id = ?`,
            [answer_id],
            (err, result) => {
                if (err) {
                    console.log(err);
                    return res.status(400).json({ error: "Something went wrong" });
                }
                return res.json({ success: "Answer deleted successfully" });
            }
        );
    } catch (error) {
        console.log(error);
        return res.status(500).json({ error: "Internal server error" });
    }
});

router.get('/get-answers-by-question', middleware, async (req, res) => {
    try {
        const { question_id } = req.query

        const questions = await executeQuery(`select ca.*, u.name from community_answers ca left join user u on ca.user_id=u.id where ca.question_id = ? and ca.status='0';`, [question_id]);

        return res.json({ success: "success", data: questions })

    } catch (error) {
        console.log(error);
        return res.status(500).json({ error: "Internal server error" })
    }
})

router.post('/verify-answer', middleware, upload.single('file'), async (req, res) => {
    try {
        const { answer_id, question, answer } = req.body;

        if (!answer_id || !question || !answer) {
            return res.status(400).json({ error: "Invalid request" });
        }


        const query = `UPDATE community_answers SET is_verified = 1 WHERE id = ?`;

        connection.execute(query, [answer_id], async (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            if (data.affectedRows === 0) {
                return res.status(404).json({ error: "Record not found" });
            }

            const addToEmbeddings = await fetch(`${BOT_URL}/add_qa`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    question,
                    answer
                })
            })

            const json = await addToEmbeddings.json();

            return res.json({ success: "Answer updated", data, addToEmbeddings: json })
        });

    } catch (error) {
        console.log(error)
        return res.status(500).json({ error: "Internal server error" });
    }
});

module.exports = router;