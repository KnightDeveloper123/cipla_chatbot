const express = require('express');
const router = express.Router();
const connection = require('../../database/db');
const executeQuery = require('../../utils/executeQuery');
const { addQuerySchema, updateQuerySchema } = require("../../validation/support");
const { middleware } = require('../../middleware/middleware');

router.post("/addQuery", middleware, async (req, res) => {
    try {
        const { query, user_id } = req.body;

        const { error } = addQuerySchema.validate(req.body, { abortEarly: false });

        if (error) {
            return res.status(400).json({ error: error.details[0]?.message });
        }

        const [checkQuery] = await executeQuery(`select * from support where query=? and status=0 and user_id=? and query_status='pending'`, [query, user_id])
        if (checkQuery) {
            return res.status(400).json({ error: "This Query already open and our team will react out to you once resolved." })
        }

        const insertQuery = 'insert into support (query, user_id, query_status) values (?, ?, ?);'
        connection.execute(insertQuery, [query, user_id, "pending"], (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            return res.json({ success: "Query Submitted", data })
        })
    } catch (error) {
        console.log("/addQuery: ", error.message);
        return res.status(500).json({ error: "Internal Server Error." });
    }
});

router.post("/updateQuery", middleware, async (req, res) => {
    try {
        const { query_id, ...rest } = req.body;
     
        const { error } = updateQuerySchema.validate(req.body, { abortEarly: false });
        if (error) {
            return res.status(400).json({ error: error.details.map(err => err.message) });
        }
        const fields = Object.keys(rest);
        const values = Object.values(rest);



        if (fields.length === 0) {
            return res.status(400).json({ error: "No valid fields provided for update." });
        }

        const setClause = fields.map(field => `${field} = ?`).join(", ");
        values.push(query_id);

        const query = `UPDATE support SET ${setClause}, updated_on = CURRENT_TIMESTAMP WHERE id = ?`;

        connection.execute(query, values, (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            if (data.affectedRows === 0) {
                return res.status(404).json({ error: "Record not found" });
            }
            return res.json({ success: "Query updated", data })
        });
    } catch (error) {
        console.error("Error in /updateQuery :", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.post("/deleteQuery", middleware, async (req, res) => {
    try {
        const { query_id } = req.body;
        console.log("Request Body:", req.body);

        // Correct condition to check if query_id is missing
        if (!query_id) {
            return res.status(400).json({ error: "Query ID is required" });
        }

        const query = `UPDATE support SET status = 1, updated_on = CURRENT_TIMESTAMP WHERE id = ?`;

        connection.execute(query, [query_id], (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" });
            }
            if (data.affectedRows === 0) {
                return res.status(404).json({ error: "Record not found" });
            }
            return res.json({ success: "Query deleted successfully!", data });
        });
    } catch (error) {
        console.error("Error in /deleteQuery:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.get("/getAllQueries", middleware, async (req, res) => {
    try {
        connection.query(`SELECT support.*, employee.name AS assignee_name
                FROM support
                LEFT JOIN employee ON support.assignee_id = employee.id
                WHERE support.status = 0`, (err, result) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            return res.json({ success: "success", data: result })
        })
    } catch (error) {
        console.error("Error in /getAllQueries:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.get("/getAllQueriesByUser", middleware, async (req, res) => {
    const { user_id } = req.query;
    try {
        connection.query(`SELECT support.*, employee.name AS assignee_name
                FROM support
                LEFT JOIN employee ON support.assignee_id = employee.id
                WHERE support.status = 0 and support.user_id=${user_id}`, (err, result) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            return res.json({ success: "success", data: result })
        })
    } catch (error) {
        console.error("Error in /getAllQueriesByUser:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.get("/getAllQueriesByEmployee", middleware, async (req, res) => {
    const { employee_id } = req.query;
    try {
        connection.query(`select * from support where status=0 and assignee_id=${employee_id}`, (err, result) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            return res.json({ success: "success", data: result })
        })
    } catch (error) {
        console.error("Error in /getAllQueriesByEmployee:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.get("/getAllQueriesById", middleware, async (req, res) => {
    const { query_id } = req.query;
    try {
        connection.query(`select * from support where status=0 and id=${query_id}`, (err, result) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            return res.json({ success: "success", data: result })
        })
    } catch (error) {
        console.error("Error in /getAllQueriesById:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

module.exports = router;